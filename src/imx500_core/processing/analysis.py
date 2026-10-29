"""Scene analysis with modlib apps only: tracking + configurable relations.

Per inference:
1. `BYTETracker`, one per class (a box nested in another class's box can't
   steal its ID), gives every detection a `tracker_id`. Tracks that only live
   on low scores for too long are ignored (static false positives).
2. `Motion` (only if a relation needs it) finds the moving regions.
3. Every relation from the project's config runs one `Matcher` per object
   class (subject ∩ object) and, with `moving: true`, one on the motion boxes.
   A subject track is *active* for a relation while all its conditions hold;
   entering/leaving that state is logged and kept in `events`.

Nothing here knows class names: they come from labels.txt and config.yaml.
Picamera2 still owns the camera; this module only sees `Detections` and a
small 3-channel copy of the frame for `Motion`.
"""

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from modlib.apps import BYTETracker, Matcher, Motion
from modlib.devices import Frame
from modlib.models import Detections

from ..config import AnalysisConfig, MatcherConfig, RelationConfig, TrackerConfig

log = logging.getLogger(__name__)

# BYTETracker scales `track_buffer` by frame_rate / 30: 30 keeps it in inferences
_TRACKER_FRAME_RATE = 30
# modlib's tracker only reads the frame size; boxes are normalized
_UNIT_FRAME = SimpleNamespace(height=1, width=1)


@dataclass
class BYTETrackerArgs:
    """Attribute bag modlib's BYTETracker expects (defined locally, as per docs)."""

    track_thresh: float
    track_buffer: int
    match_thresh: float
    mot20: bool
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0

    @classmethod
    def from_config(cls, cfg: TrackerConfig) -> "BYTETrackerArgs":
        return cls(cfg.track_thresh, cfg.track_buffer, cfg.match_thresh, cfg.mot20)


@dataclass(frozen=True)
class RelationEvent:
    """A subject track entered (`active=True`) or left a relation."""

    relation: str
    subject: str
    track_id: int
    active: bool


def empty_detections() -> Detections:
    dets = Detections(
        bbox=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.uint16),
    )
    dets.tracker_id = np.empty((0,), dtype=int)
    return dets


def concat(parts: list[Detections]) -> Detections:
    """Concatenates tracked Detections (older modlib has no `__add__`)."""
    parts = [p for p in parts if len(p)]
    if not parts:
        return empty_detections()
    out = Detections(
        bbox=np.concatenate([p.bbox for p in parts]),
        confidence=np.concatenate([p.confidence for p in parts]),
        class_id=np.concatenate([p.class_id for p in parts]),
    )
    out.tracker_id = np.concatenate([np.asarray(p.tracker_id) for p in parts])
    return out


def _matcher(cfg: MatcherConfig) -> Matcher:
    return Matcher(
        max_missing_overlap=cfg.max_missing_overlap,
        max_missing_tracker=cfg.max_missing_tracker,
        min_overlap_threshold=cfg.min_overlap_threshold,
        hysteresis=cfg.hysteresis,
    )


def _ids(dets: Detections) -> set[int]:
    return {int(t) for t in dets.tracker_id} if len(dets) else set()


def _class_id(labels: list[str], name: str, relation: str) -> int:
    if name not in labels:
        raise ValueError(f"Relation '{relation}': class '{name}' not in {labels}")
    return labels.index(name)


@dataclass
class Relation:
    """Runtime state of one `RelationConfig`."""

    cfg: RelationConfig
    subject_id: int
    object_ids: dict[str, int]
    object_matchers: dict[str, Matcher]
    motion_matcher: Matcher | None
    with_object: dict[str, set[int]] = field(default_factory=dict)
    moving: set[int] = field(default_factory=set)
    active: set[int] = field(default_factory=set)

    @classmethod
    def build(cls, cfg: RelationConfig, labels: list[str]) -> "Relation":
        return cls(
            cfg=cfg,
            subject_id=_class_id(labels, cfg.subject, cfg.name),
            object_ids={o: _class_id(labels, o, cfg.name) for o in cfg.objects},
            object_matchers={o: _matcher(cfg.matcher) for o in cfg.objects},
            motion_matcher=_matcher(cfg.motion_matcher) if cfg.moving else None,
        )

    def update(
        self, tracks: Detections, motion_boxes: Detections | None
    ) -> list[RelationEvent]:
        subjects = tracks[tracks.class_id == self.subject_id]
        conditions = []

        for name, class_id in self.object_ids.items():
            objects = tracks[tracks.class_id == class_id]
            matched = subjects[self.object_matchers[name].match(subjects, objects)]
            self.with_object[name] = _ids(matched)
            conditions.append(self.with_object[name])

        if self.motion_matcher is not None:
            boxes = motion_boxes if motion_boxes is not None else empty_detections()
            self.moving = _ids(subjects[self.motion_matcher.match(subjects, boxes)])
            conditions.append(self.moving)

        active = set.intersection(*conditions)
        events = [
            RelationEvent(self.cfg.name, self.cfg.subject, t, True)
            for t in sorted(active - self.active)
        ] + [
            RelationEvent(self.cfg.name, self.cfg.subject, t, False)
            for t in sorted(self.active - active)
        ]
        self.active = active
        return events

    def tags(self, track_id: int) -> list[str]:
        """Conditions this track meets, for its label (`+package`, `moving`)."""
        out = [f"+{o}" for o, ids in self.with_object.items() if track_id in ids]
        if track_id in self.moving:
            out.append("moving")
        return out


class SceneAnalyzer:
    def __init__(self, labels: list[str], cfg: AnalysisConfig) -> None:
        self._labels = labels
        self._cfg = cfg
        self.relations = [Relation.build(r, labels) for r in cfg.relations]

        args = BYTETrackerArgs.from_config(cfg.tracker)
        self._trackers = {
            c: BYTETracker(args, frame_rate=_TRACKER_FRAME_RATE)
            for c in range(len(labels))
        }

        m = cfg.motion
        self._motion = (
            Motion(size_threshold=max(1, round(m.size_threshold * m.scale**2)))
            if cfg.needs_motion
            else None
        )

        # Inference index of each track's last high-score (> track_thresh) box
        self._last_high: dict[int, int] = {}
        self._inference = 0

        self.tracks = empty_detections()
        self.motion_boxes = empty_detections()
        self.events: list[RelationEvent] = []

    @property
    def needs_motion(self) -> bool:
        return self._motion is not None

    @property
    def active_ids(self) -> set[int]:
        """Tracks active in at least one relation."""
        return set().union(*(r.active for r in self.relations))

    def update(self, detections: Detections, motion_frame: Frame | None) -> None:
        """One inference: track, detect motion if needed, evaluate relations."""
        self.tracks = self._track(detections)

        motion = None
        if self._motion is not None and motion_frame is not None:
            self.motion_boxes = motion = self._motion.detect(motion_frame)

        self.events = [e for r in self.relations for e in r.update(self.tracks, motion)]
        for e in self.events:
            log.info(
                "%s: %s #%d %s",
                e.relation,
                e.subject,
                e.track_id,
                "started" if e.active else "ended",
            )

    def _track(self, detections: Detections) -> Detections:
        self._inference += 1
        tracks = self._confirmed(self._bytetrack(detections))
        self._forget_old_tracks()
        return tracks

    def _bytetrack(self, detections: Detections) -> Detections:
        parts = []
        # Every class is updated, even without detections, so its tracks age
        for class_id, tracker in self._trackers.items():
            dets = detections[detections.class_id == class_id]
            dets = tracker.update(_UNIT_FRAME, dets)
            if len(dets):
                parts.append(dets[np.asarray(dets.tracker_id) >= 0])
        return concat(parts)

    def _confirmed(self, tracks: Detections) -> Detections:
        """Drops tracks living only on low scores for > `max_low_streak`.

        ByteTrack keeps a track alive with low-score boxes. A static false
        positive (a door at 0.38) overlapping where a real person stood
        inherits that person's track forever. A track must see a high score
        regularly; once dropped, the false positive alone can't revive it.
        """
        if not len(tracks):
            return tracks
        cfg = self._cfg.tracker
        ids = np.asarray(tracks.tracker_id)
        for tid, score in zip(ids, tracks.confidence):
            if score > cfg.track_thresh:
                self._last_high[int(tid)] = self._inference
        age = np.array(
            [self._inference - self._last_high.get(int(t), -(10**9)) for t in ids]
        )
        return tracks[age <= cfg.max_low_streak]

    def _forget_old_tracks(self) -> None:
        horizon = self._cfg.tracker.max_low_streak + self._cfg.tracker.track_buffer
        self._last_high = {
            t: i for t, i in self._last_high.items() if self._inference - i <= horizon
        }

    def labels_for(self, dets: Detections) -> list[str]:
        out = []
        for _, score, cls, tid in dets:
            name = self._labels[cls] if cls < len(self._labels) else str(cls)
            tags = []
            for r in self.relations:
                if cls == r.subject_id:
                    tags += [t for t in r.tags(int(tid)) if t not in tags]
            tag = "".join(f" {t}" for t in tags)
            out.append(f"#{tid} {name}{tag} {score:.2f}")
        return out

    def summary(self) -> str:
        n = {c: int((self.tracks.class_id == c).sum()) for c in self._trackers}
        parts = [", ".join(f"{self._labels[c]}={k}" for c, k in n.items())]
        parts += [f"{r.cfg.name}: {sorted(r.active)}" for r in self.relations]
        return "tracks: " + " | ".join(parts)
