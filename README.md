# IMX500 AI Camera Applications

Modular Computer Vision and Edge AI monorepo optimized for the **Raspberry Pi AI Camera (Sony IMX500 sensor)**. Inference runs on the IMX500 itself; `Picamera2` owns the camera, its buffers and the sensor controls; `modlib` is used only as an API for post-processing, tracking, matching and annotation; **MediaMTX** serves low-latency RTSP/WebRTC video.

## 🌟 Key Features

* **On-Sensor Inference:** Deep Learning models (packaged as `.rpk`) run on the IMX500's integrated AI accelerator, so the host CPU never runs the network. H.264 encoding is done by the hardware encoder of the VideoCore GPU (Pi Zero 2 W), leaving the CPU for post-processing, tracking, motion detection, drawing and RTSP packaging.

* **Picamera2 + modlib Hybrid:** `Picamera2` manages capture and buffers, while `modlib` provides post-processing (e.g., `pp_od_yolo_ultralytics`), tracking (`BYTETracker`), spatial relations (`Matcher`), motion (`Motion`) and annotation (`Annotator`). modlib's own `AiCamera` is not used: it allocates fixed-size buffers that don't fit in the Pi Zero 2 W's RAM (the hybrid runs at ~165 MB RSS).

* **Monorepo Architecture:** Centralized shared core (`src/imx_core`) with isolated, task-specific applications in `projects/`.

* **Lightweight Web Dashboard:** Standard-library HTTP server (no web framework) with hardware telemetry (SoC temperature, RAM, throttling), event history with clips, and a live WebRTC preview.

* **Low-Latency Video Streaming:** Integrated **MediaMTX** pipeline for RTSP (TCP) and WebRTC.

* **Deployment & Tooling:** `systemd` services with automatic restart (the app exits on fatal errors or when the camera freezes, so systemd recovers it), `uv` dependency management, `Ruff` linting/formatting and a `pytest` suite (hardware mocked).

## 🛠️ Hardware & Operating System Requirements

* **Device:** Tested on Raspberry Pi Zero 2 W.

* **Camera:** Raspberry Pi AI Camera with **Sony IMX500 sensor**. Adding a heatsink or some airflow is recommended: without it, the IMX500 can overheat and the camera times out.

* **OS:** Raspberry Pi OS Lite 64-bit **Trixie (Debian 13)**, which ships Python 3.13 (required by `pyproject.toml`; uv is configured to use only the system interpreter).

* **System Dependencies:** `imx500-all`, `python3-picamera2`, `python3-opencv` (from apt) and [MediaMTX](https://github.com/bluenviron/mediamtx/releases) (`linux_arm64` binary).

* **Models:** IMX500 `.rpk` packages are not tracked in git (`*.rpk` is ignored). Export your model for the IMX500 (e.g. Ultralytics `format="imx"`), package it with `imx500-package` and place it where the project's config expects it.

## 📂 Repository Layout

```
imx500_ai_camera_applications/
├── deploy/                 # Production deployment scripts
│   ├── mediamtx/           # MediaMTX configuration (mediamtx.yml)
│   └── systemd/            # Systemd service units (mediamtx & imx500-app)
├── projects/               # Isolated applications by task / dataset
│   ├── porch_piracy/       # Example: Delivery detection task
│   └── strawberry_disease/ # Example: Agricultural monitoring task
├── src/
│   └── imx_core/           # Shared core library
│       ├── camera/         # Picamera2 hardware encapsulation
│       ├── monitors/       # Local dashboard & hardware telemetry
│       ├── processing/     # modlib post-processing, tracking, matching & annotation
│       └── services/       # External notification dispatchers (MQTT, Telegram; planned)
├── tests/                  # Automated pytest suite
├── Makefile                # Command shortcuts for setup, testing, and deployment
└── pyproject.toml          # Project dependencies and tool configurations (uv)
```

## ⚙️ Quick Start & Installation

### 1. Prerequisites

Update the Raspberry Pi and install the camera stack:

```bash
sudo apt update && sudo apt full-upgrade
sudo apt install imx500-all python3-picamera2 python3-opencv --no-install-recommends
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Download MediaMTX into your home folder (pick the latest `linux_arm64` release):

```bash
mkdir -p ~/mediamtx && cd ~/mediamtx
wget https://github.com/bluenviron/mediamtx/releases/download/v1.21.2/mediamtx_v1.21.2_linux_arm64.tar.gz
tar -xzf mediamtx_v1.21.2_linux_arm64.tar.gz
```

### 2. Clone and Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Gluwxy/imx500_ai_camera_applications.git
cd imx500_ai_camera_applications
```

Create the virtual environment with access to system packages (`picamera2`, `libcamera`, `numpy` and `cv2` come from the OS, not from PyPI):

```bash
uv venv --system-site-packages --python /usr/bin/python3
```

Synchronize the dependencies (including `modlib`, which brings `lap` for the tracker):

```bash
make sync
```

### 3. Environment Configuration

Copy the environment variables template and fill in your secrets:

```bash
cp .env.example .env
```

## 🚀 Running Applications

Each project in `projects/` has its own README with how it works, its model, configuration and known limitations.

| Project | Task | Model |
| ------- | ---- | ----- |
| [Porch Piracy](projects/porch_piracy/README.md) | Detects people taking packages from the porch | YOLO11n (`package`, `person`) |

<!-- GIF: porch_piracy demo -->

Example with `porch_piracy`. Models are not stored in git, so download the project's model from GitHub Releases first, then run it:

```bash
make model PROJECT=porch_piracy
make run PROJECT=porch_piracy
```

The dashboard starts automatically on `http://<raspberry-pi-ip>:8000` with live telemetry, the stream preview and the event history.

Watch the RTSP stream from another machine:

```bash
ffplay -rtsp_transport tcp rtsp://<raspberry-pi-ip>:8554/<stream-path>
```

## 🔧 Production Deployment with Systemd & MediaMTX

Install MediaMTX and the application as services that start on boot and restart on failure:

```bash
# Install and enable systemd services
make setup-services

# Start the services
make start-services

# View live system logs
make logs
```

## 🧪 Quality Assurance & Testing

Run code quality checks and unit tests:

```bash
# Run unit tests
make test

# Check linting and formatting rules with Ruff
make lint

# Automatically format code
make format
```

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.