# IMX500 AI Camera Applications
# Usage: make <target> [PROJECT=porch_piracy]   (`make help` lists targets)

REPO        ?= Gluwxy/imx500_ai_camera_applications
PROJECT     ?= porch_piracy
PROJECT_DIR := $(firstword $(wildcard projects/*/$(PROJECT)))
PYTHON      ?= /usr/bin/python3
VENV_PY     := .venv/bin/python

# Model declared by the project (MODEL_TAG, MODEL_DIR, MODEL_FILES)
-include $(PROJECT_DIR)/project.mk
RELEASE_URL = https://github.com/$(REPO)/releases/download/$(MODEL_TAG)

.DEFAULT_GOAL := help
.PHONY: help sync setup-pi lint format test check project-check model model-check \
	run mediamtx setup-services start-services stop-services restart status logs

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "} {printf "  %-16s %s\n", $$1, $$2}'

# --- Environment -------------------------------------------------------------

sync: ## Install dependencies from uv.lock (dev machine / CI)
	uv sync --locked

setup-pi: ## Pi only: venv with system picamera2/libcamera, then sync
	test -d .venv || uv venv --system-site-packages --python $(PYTHON)
	uv sync --locked

# --- Quality -----------------------------------------------------------------

lint: ## Ruff lint + format check (same as CI)
	uv run ruff check .
	uv run ruff format --check .

format: ## Fix lint issues and format the code
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite
	uv run pytest

check: lint test ## Everything CI runs

# --- Models ------------------------------------------------------------------

project-check:
	@test -n "$(PROJECT_DIR)" || { echo "Project '$(PROJECT)' not found in projects/*/"; exit 1; }

model: project-check ## Download PROJECT's model from GitHub Releases and verify it
	@test -n "$(MODEL_TAG)" || { echo "$(PROJECT_DIR)/project.mk: MODEL_TAG not set"; exit 1; }
	@mkdir -p $(MODEL_DIR)
	@for f in $(MODEL_FILES); do \
		echo "Downloading $$f ($(MODEL_TAG))"; \
		curl -fL --retry 3 -o $(MODEL_DIR)/$$f $(RELEASE_URL)/$$f || exit 1; \
	done
	@cd $(MODEL_DIR) && sha256sum -c SHA256SUMS

model-check: project-check
	@test -f $(MODEL_DIR)/network.rpk || { \
		echo "Model not found in $(MODEL_DIR). Run: make model PROJECT=$(PROJECT)"; exit 1; }

# --- Run on the Pi -----------------------------------------------------------

run: model-check ## Run PROJECT in the foreground (stop the service first)
	$(VENV_PY) -m imx500_core $(PROJECT)

mediamtx: ## Run MediaMTX by hand with the repo config
	cd deploy/mediamtx && ~/mediamtx/mediamtx mediamtx.yml

# --- systemd -----------------------------------------------------------------

setup-services: ## Install and enable the systemd units
	sudo cp deploy/systemd/mediamtx.service deploy/systemd/imx500_app.service /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable mediamtx.service imx500_app.service
	@echo "Services installed. Project: edit PROJECT in imx500_app.service (sudo systemctl edit imx500_app)."

start-services: ## Start MediaMTX and the app
	sudo systemctl start mediamtx.service imx500_app.service

stop-services: ## Stop the app and MediaMTX
	sudo systemctl stop imx500_app.service mediamtx.service

restart: ## Restart the app
	sudo systemctl restart imx500_app.service

status: ## Show both services
	systemctl --no-pager status mediamtx.service imx500_app.service

logs: ## Follow the app logs (-a: the firmware progress bar prints \r)
	journalctl -a -u imx500_app.service -f
