# AGENTS.md

Monorepo of Computer Vision / Edge AI applications for the **Raspberry Pi AI Camera (Sony IMX500)** on a **Raspberry Pi Zero 2 W**. Inference runs **on the IMX500 sensor**; the Pi parses the outputs, tracks, relates objects, draws, encodes H.264 (hardware) and streams through MediaMTX (RTSP/WebRTC).

First project: **porch piracy** (`projects/object_detection/porch_piracy`), a YOLO11n with classes `package` and `person` that flags a person carrying a package away. It is being ported from the previous repo `Gluwxy/raspberry_pi_ai_camera_rtsp` (package `imx500_rtsp`, branches `feat/modlib-tracker`, `feat/modlib-annotator`).

## Commands

```bash
# Anywhere (tests mock all the hardware)
uv sync                                   # .venv with the dev group (pytest, ruff, pre-commit)
uv run pytest                             # same as CI
uv run ruff check --fix && uv run ruff format
uv run pre-commit install                 # Ruff on every commit (uses the locked Ruff)
uv run pre-commit run pytest --hook-stage manual

# Dependencies: edit pyproject.toml, then
uv lock                                   # commit pyproject.toml + uv.lock together (CI runs `uv sync --locked`)

# Models (not in git)
make model PROJECT=porch_piracy           # download network.rpk from GitHub Releases (planned target)
```

- Python **3.13** only (`>=3.13,<3.14`): Raspberry Pi OS Trixie ships 3.13 and modlib supports `< 3.14`. uv never downloads an interpreter (`python-downloads = "never"`).
- On the Pi the venv uses `--system-site-packages`: `picamera2` and `libcamera` come from the OS (`python3-picamera2`). modlib pulls `numpy`, `opencv-python` and `lapx` into the venv.
- In systemd use `.venv/bin/python -m imx500_core`, not `uv run` (it forwards Ctrl+C, so SIGINT arrives twice).
- App logs contain `\r` from the IMX500 firmware progress bar: use `grep -a`.
- MediaMTX must be started from the folder that holds its `mediamtx.yml`.

## Layout

- `src/imx500_core/` — shared core (`module-name` in `pyproject.toml`; the project name differs).
  - `config.py` — frozen Pydantic models (`extra="forbid"`: a new YAML key needs a model field). `load_config("porch_piracy")` merges the root `config.yaml` (shared) with `projects/<task>/<project>/config.yaml` (wins key by key); model paths are relative to the project folder; secrets (`TELEGRAM_TOKEN`, `MQTT_PASSWORD`) come from the environment or the root `.env`.
  - `camera/camera.py` — IMX500 + Picamera2 setup (`setup_camera`); closes the camera if configuration fails.
  - `main.py` / `__main__.py` — entry point (stub).
  - Planned: `processing/` (modlib pp, tracker, matcher, motion, annotator), `monitors/` (dashboard, telemetry), `services/` (MQTT, Telegram).
- `projects/<task>/<project>/` — one app per folder: `config.yaml`, `labels.txt`, `models/` (git-ignored `.rpk`), README.
- `tests/` — pytest. `conftest.py` mocks `picamera2` when it isn't installed and creates fake model files in `tmp_path` (`*.rpk` is git-ignored, never commit a dummy one). `tests/fixtures/` holds a base config and a project config.
- `deploy/` — systemd units and MediaMTX config (to be ported).
- `.github/workflows/ci.yml` — Python 3.13, `uv sync --locked`, Ruff check + format, pytest.

## Rules

- **Low resources first**: Pi Zero 2 W, ~415 MB usable RAM. Prefer the stdlib; no web frameworks or heavy deps.
- **Picamera2 owns the camera, modlib is only an API.** modlib's `AiCamera` allocates fixed-size buffers that don't fit in the Pi Zero's RAM. Use modlib's pure-numpy pieces (`pp_od_yolo_ultralytics`, `Annotator`, `BYTETracker`, `Matcher`, `Motion`) on Picamera2's `MappedArray` (~165 MB RSS measured).
- Everything after opening the camera goes **inside the `try`** in `main()` so `finally` always releases it; each cleanup step runs on its own so one failure doesn't skip the rest.
- The `pre_callback` runs on the camera thread for every frame: keep it fast and never let it raise. Anything slow (writing clips, fsync, network) goes to a worker thread.
- Keep tracker / relation logic scenario-agnostic: no class names in Python; they come from `labels.txt` and the project config.
- Never call blocking `capture_metadata()` in the main loop; read state from the overlay/callback.
- Exit code 1 on fatal errors/watchdog → systemd (`Restart=always`) recovers the camera.
- Behaviour changes come with a test; when fixing a field bug, build the test from the Pi logs.
- Code, comments and UI in English. Brief conventional commits (`feat:`, `fix:`, `test:`, `refactor:`, `ci:`, `chore:`, `docs:`).
- Never commit `.env`, `*.rpk`, `*.crt`, `*.key`, `data/`, datasets.

## Known facts (learned the hard way)

**Model & tracking**
- IMX500 scores are quantized to sigmoid steps: 0.27, 0.38, 0.50, 0.62, 0.73, 0.82, 0.88, 0.93, 0.95. A threshold really picks a level.
- Fully visible people score ~0.85; a person holding a box scores ~0.27–0.50 (torso hidden).
- Ultralytics exports include NMS; outputs are boxes, scores, classes, count (count has shape `(1,)`: newer numpy refuses `int()` on it). Outputs can be `None` on some frames.
- Static false positives exist: a door scored as `person` at 0.38. ByteTrack keeps tracks alive with low scores, so the door **inherited the track of a real person** who stood in front of it. Fix: keep `min_score` low (0.25, low scores keep tracks), raise `track_thresh` (0.45 → new tracks need > 0.55) and drop tracks that see no high score for `max_low_streak` inferences. Raising `min_score` instead breaks tracking under occlusion.
- Use one modlib `BYTETracker` per class with `mot20=True` (the default fuses the score into the cost and drops carried packages). ByteTrack has no re-ID: re-entering the scene gives a new ID.
- modlib `Matcher` only reports a pair after > 20 matched inferences (~2 s at 10 FPS) and releases it slowly (60-inference average). Its overlap is intersection / area of the *matched* box.
- modlib `Motion` uses `cv2.subtract` (only brightening changes), its `motion_threshold` is unused, and it expects a 3-channel BGR image: feed it a downscaled 3-channel copy, computed **before** drawing boxes. It also reacts to pets/shadows; only motion inside a person counts.
- The on-sensor NMS still lets duplicate boxes of one person through → per-class NMS before the tracker.
- A person filling the frame has no ΔY: events need the subject to walk toward/away from the camera.

**Image & coordinates**
- `XBGR8888` is `[R, G, B, X]` in numpy (`BGR888` is `[R, G, B]`): modlib draws with `Color.as_bgr()`, so swap R/B in its palette for those formats.
- Inference coords are normalized to the input tensor (full sensor unless an inference ROI is set); the 640×640 output is a centered square `ScalerCrop`. Express the mapping as a modlib `ROI` and apply it once per inference so detections, tracks and motion share image space.

**Video & clips**
- H.264 is encoded by the VideoCore hardware encoder (Pi Zero 2 W / Pi 4); the Pi 5 has none (software encoding on the CPU).
- `CircularOutput2` writes frames as they age out of its buffer: a 5 s + 5 s clip must stay open `buffer + post` = 10 s.
- PyAV writes an `avcC` whose SPS/PPS don't match the Pi's hardware encoder: Chrome/FFmpeg play it, Safari/QuickTime show black → rewrite the `avcC` from the in-band SPS/PPS.
- Safari needs HTTP Range (206) to play video; `SimpleHTTPRequestHandler` lacks it.

**Hardware**
- The IMX500 overheats without cooling: `SensorTemperature` saturates at 80 °C, then "Camera frontend has timed out". Lower FPS only delays it; real fix is a heatsink/airflow.
- The ~16 FPS inference ceiling is compute-bound (fixed accelerator clock), not thermal.
- CPU and GPU share one SoC thermal sensor; there is no GPU utilisation metric on the Pi.
- Swap is zram (in RAM); `rpi-zram-writeback.timer` may write idle pages to `/var/swap` on the SD.
