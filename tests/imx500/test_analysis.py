"""SceneAnalyzer: modlib BYTETracker + Matcher + Motion on synthetic scenes."""

import logging

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("modlib")
pytest.importorskip("lap")
pytest.importorskip("cv2")

from modlib.devices import IMAGE_TYPE, Frame  # noqa: E402
from modlib.models import ROI, Detections  # noqa: E402

from imx500_core.config import (  # noqa: E402
    AnalysisConfig,
    RelationConfig,
    TrackerConfig,
)
from imx500_core.processing.analysis import (  # noqa: E402
    SceneAnalyzer,
    concat,
    empty_detections,
)

LABELS = ["package", "person", "bag"]
PACKAGE, PERSON, BAG = 0, 1, 2
SIZE = 320  # motion frame side (px)
# Matcher only reports objects matched for more than 20 inferences
WARMUP = 25

PERSON_BOX = (0.30, 0.10, 0.60, 0.70)
FAR_BOX = (0.80, 0.80, 0.95, 0.95)


def _relation(name="carrying", subject="person", objects=("package",), moving=True):
    return RelationConfig(name=name, subject=subject, objects=objects, moving=moving)


def _analyzer(*relations, **tracker) -> SceneAnalyzer:
    cfg = AnalysisConfig(tracker=TrackerConfig(**tracker), relations=relations)
    return SceneAnalyzer(LABELS, cfg)


def _dets(*boxes) -> Detections:
    """(class_id, score, x1, y1, x2, y2) normalized to the image."""
    if not boxes:
        return empty_detections()
    arr = np.array(boxes, dtype=np.float32)
    return Detections(
        bbox=arr[:, 2:6].copy(),
        confidence=arr[:, 1].copy(),
        class_id=arr[:, 0].astype(np.uint16),
    )


def _frame(moving_box, step: int) -> Frame:
    """Gray background; `moving_box` is a textured patch that changes each step."""
    img = np.full((SIZE, SIZE, 3), 40, np.uint8)
    if moving_box is not None:
        x1, y1, x2, y2 = (int(v * SIZE) for v in moving_box)
        rng = np.random.default_rng(step)
        img[y1:y2, x1:x2] = rng.integers(0, 255, (y2 - y1, x2 - x1, 1), np.uint8)
    return Frame(
        timestamp="",
        image=img,
        image_type=IMAGE_TYPE.VGA,
        width=SIZE,
        height=SIZE,
        channels=3,
        detections=None,
        new_detection=True,
        fps=0.0,
        dps=0.0,
        roi=ROI(0.0, 0.0, 1.0, 1.0),
    )


def _walk(a, n, *, carrying=True, moving=True, start=0, extra=()):
    """Person walking down the image; the package inside it (or far away)."""
    for i in range(start, start + n):
        dy = 0.004 * i if moving else 0.0
        person = (0.30, 0.10 + dy, 0.60, 0.70 + dy)
        package = (0.38, 0.40 + dy, 0.52, 0.55 + dy) if carrying else FAR_BOX
        dets = _dets((PERSON, 0.85, *person), (PACKAGE, 0.62, *package), *extra)
        a.update(dets, _frame(person if moving else None, i))


def _ids(a, class_id) -> set[int]:
    t = a.tracks
    return {int(i) for i in t.tracker_id[t.class_id == class_id]}


# --- Person carrying a package while moving (porch_piracy) --------------------


def test_moving_person_with_package_becomes_active(caplog):
    a = _analyzer(_relation())
    with caplog.at_level(logging.INFO, logger="imx500_core.processing.analysis"):
        _walk(a, WARMUP)

    (pid,) = _ids(a, PERSON)
    (rel,) = a.relations
    assert rel.with_object == {"package": {pid}}
    assert rel.moving == {pid}
    assert a.active_ids == {pid}
    assert f"carrying: person #{pid} started" in caplog.text
    assert caplog.text.count("started") == 1  # once, not every inference


def test_still_person_with_package_is_not_active():
    a = _analyzer(_relation())
    _walk(a, WARMUP, moving=False)
    (pid,) = _ids(a, PERSON)
    (rel,) = a.relations
    assert rel.with_object["package"] == {pid}
    assert rel.moving == set()
    assert a.active_ids == set()


def test_package_outside_the_person_is_not_matched():
    a = _analyzer(_relation())
    _walk(a, WARMUP, carrying=False)
    assert a.relations[0].with_object["package"] == set()
    assert a.active_ids == set()


def test_leaving_the_package_ends_the_relation(caplog):
    a = _analyzer(_relation(moving=False))
    _walk(a, WARMUP)
    (pid,) = _ids(a, PERSON)
    with caplog.at_level(logging.INFO, logger="imx500_core.processing.analysis"):
        for i in range(WARMUP, WARMUP + 80):
            person = (0.30, 0.10 + 0.002 * i, 0.60, 0.70 + 0.002 * i)
            a.update(_dets((PERSON, 0.85, *person), (PACKAGE, 0.62, *FAR_BOX)), None)
    assert a.active_ids == set()
    assert f"carrying: person #{pid} ended" in caplog.text


# --- Generic relations ---------------------------------------------------------


def test_objects_only_relation_does_not_run_motion():
    a = _analyzer(_relation(moving=False))
    assert not a.needs_motion
    _walk(a, WARMUP, moving=False)
    (pid,) = _ids(a, PERSON)
    assert a.active_ids == {pid}
    assert len(a.motion_boxes) == 0


def test_motion_only_relation_on_any_class():
    """`moving: true` without objects: a moving package, the person still."""
    a = _analyzer(_relation(name="moving_package", subject="package", objects=()))
    assert a.needs_motion
    for i in range(WARMUP):
        box = (0.10, 0.10 + 0.004 * i, 0.30, 0.30 + 0.004 * i)
        dets = _dets((PACKAGE, 0.88, *box), (PERSON, 0.85, 0.6, 0.5, 0.9, 0.95))
        a.update(dets, _frame(box, i))
    (pkg,) = _ids(a, PACKAGE)
    assert a.active_ids == {pkg}
    assert a.labels_for(a.tracks[a.tracks.class_id == PACKAGE])[0].startswith(
        f"#{pkg} package moving"
    )


def test_every_object_must_overlap_the_subject():
    rel = _relation(name="loaded", objects=("package", "bag"), moving=False)
    a = _analyzer(rel)
    bag_inside = (BAG, 0.88, 0.40, 0.45, 0.50, 0.60)
    _walk(a, WARMUP, moving=False)  # package only
    assert a.active_ids == set()
    _walk(a, WARMUP, moving=False, start=WARMUP, extra=(bag_inside,))
    (pid,) = _ids(a, PERSON)
    assert a.active_ids == {pid}


def test_relations_are_independent_and_tags_merge():
    with_package = _relation(name="with_package", moving=False)
    moving_person = _relation(name="moving_person", objects=(), moving=True)
    a = _analyzer(with_package, moving_person)
    _walk(a, WARMUP)
    (pid,) = _ids(a, PERSON)
    assert [r.active for r in a.relations] == [{pid}, {pid}]
    person = a.tracks[a.tracks.class_id == PERSON]
    assert a.labels_for(person) == [f"#{pid} person +package moving 0.85"]
    assert "with_package: [" in a.summary() and "moving_person: [" in a.summary()


# --- Tracker -------------------------------------------------------------------

# Field case (Pi, door scene): a person (0.85) takes the package and leaves;
# the door behind them is a static person false positive at 0.38.
DOOR = (0.15, 0.0, 0.75, 1.0)
BOX = (0.35, 0.85, 0.60, 1.0)


def _step(a, *dets):
    a.update(_dets(*dets), None)


def test_door_false_positive_does_not_inherit_the_person_track():
    a = _analyzer(_relation(moving=False))
    for _ in range(10):
        _step(a, (PERSON, 0.85, 0.18, 0.02, 0.72, 1.0), (PACKAGE, 0.93, *BOX))
    (pid,) = _ids(a, PERSON)
    for _ in range(40):
        _step(a, (PERSON, 0.38, *DOOR), (PACKAGE, 0.93, *BOX))
    assert _ids(a, PERSON) == set()
    assert a.active_ids == set()
    for _ in range(10):
        _step(a, (PERSON, 0.38, *DOOR))
    for _ in range(30):
        _step(a, (PERSON, 0.38, *DOOR), (PACKAGE, 0.88, *BOX))
    assert _ids(a, PERSON) == set()
    assert a.active_ids == set()


def test_short_low_score_dip_keeps_the_same_track():
    a = _analyzer()
    for _ in range(5):
        _step(a, (PERSON, 0.85, *PERSON_BOX))
    (pid,) = _ids(a, PERSON)
    for _ in range(10):  # < max_low_streak
        _step(a, (PERSON, 0.38, *PERSON_BOX))
        assert _ids(a, PERSON) == {pid}
    _step(a, (PERSON, 0.85, *PERSON_BOX))
    assert _ids(a, PERSON) == {pid}


def test_low_streak_is_configurable():
    a = _analyzer(max_low_streak=3)
    _step(a, (PERSON, 0.85, *PERSON_BOX))
    for _ in range(3):
        _step(a, (PERSON, 0.38, *PERSON_BOX))
    assert len(_ids(a, PERSON)) == 1
    _step(a, (PERSON, 0.38, *PERSON_BOX))
    assert _ids(a, PERSON) == set()


def test_nested_box_keeps_its_own_track():
    """One tracker per class: the package inside the person gets another ID."""
    a = _analyzer()
    _walk(a, 5)
    assert len(set(a.tracks.tracker_id.tolist())) == 2


# --- Config validation -----------------------------------------------------------


def test_unknown_class_fails_fast():
    with pytest.raises(ValueError, match="car"):
        _analyzer(_relation(objects=("car",)))


@pytest.mark.parametrize(
    ("relations", "message"),
    [
        ([{"name": "x", "subject": "person"}], "needs `objects` or `moving`"),
        (
            [{"name": "x", "subject": "person", "objects": ["person"]}],
            "subject is also an object",
        ),
        (
            [
                {"name": "x", "subject": "person", "moving": True},
                {"name": "x", "subject": "package", "moving": True},
            ],
            "unique",
        ),
    ],
)
def test_invalid_relations_are_rejected(relations, message):
    with pytest.raises(ValidationError, match=message):
        AnalysisConfig.model_validate({"relations": relations})


def test_concat_keeps_tracker_ids():
    a, b = _dets((1, 0.5, 0, 0, 0.1, 0.1)), _dets((0, 0.6, 0.2, 0.2, 0.3, 0.3))
    a.tracker_id, b.tracker_id = np.array([3]), np.array([7])
    out = concat([a, empty_detections(), b])
    assert out.tracker_id.tolist() == [3, 7]
    assert out.class_id.tolist() == [1, 0]
    assert len(concat([])) == 0


# --- Events ---------------------------------------------------------------------


def _with(detections=(), alerts=(), *relations):
    cfg = AnalysisConfig(relations=relations, detections=detections, alerts=alerts)
    return SceneAnalyzer(LABELS, cfg)


def test_detection_events_start_and_end_after_track_buffer():
    a = _with(("person",))
    events = []
    a.on_event = events.append
    for _ in range(3):
        _step(a, (PERSON, 0.85, *PERSON_BOX), (PACKAGE, 0.93, *BOX))
    (pid,) = _ids(a, PERSON)
    assert [(e.kind, e.name, e.track_id, e.active) for e in events] == [
        ("detection", "person", pid, True)
    ]  # once, and only for the configured classes
    for _ in range(31):  # track_buffer (30) inferences without the person
        _step(a)
    assert events[-1].name == "person" and not events[-1].active
    assert len(events) == 2


def test_relation_events_reach_on_event_with_json():
    a = _with((), (), _relation(moving=False))
    events = []
    a.on_event = events.append
    _walk(a, WARMUP, moving=False)
    (event,) = events
    data = event.json()
    assert data["kind"] == "relation" and data["name"] == "carrying"
    assert data["subject"] == "person" and data["active"] is True
    assert "T" in data["time"]  # ISO 8601


def test_failing_event_handler_does_not_break_the_analysis(caplog):
    a = _with(("person",))
    a.on_event = lambda e: 1 / 0
    _step(a, (PERSON, 0.85, *PERSON_BOX))
    assert len(_ids(a, PERSON)) == 1
    assert "Event handler failed" in caplog.text


def test_alert_ids_only_cover_alert_relations():
    plain = _relation(name="with_package", moving=False)
    alert = _relation(name="moving_person", objects=(), moving=True)
    a = _with((), ("moving_person",), plain, alert)
    assert a.alert_names == ["moving_person"]
    _walk(a, WARMUP)
    (pid,) = _ids(a, PERSON)
    assert a.active_ids == {pid} and a.alert_ids == {pid}


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        ({"alerts": ["ghost"]}, "alerts not in detections/relations"),
        (
            {
                "detections": ["person"],
                "relations": [{"name": "person", "subject": "bag", "moving": True}],
            },
            "clash",
        ),
    ],
)
def test_invalid_events_are_rejected(cfg, message):
    with pytest.raises(ValidationError, match=message):
        AnalysisConfig.model_validate(cfg)


def test_unknown_detection_class_fails_fast():
    with pytest.raises(ValueError, match="car"):
        _with(("car",))
