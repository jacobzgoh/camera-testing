#!/usr/bin/env sh
set -eu

mkdir -p models

PT_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
PT_PATH="models/yolov8n-seg.pt"
MODEL_PATH="models/yolov8n-seg.tflite"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if [ -s "$MODEL_PATH" ] && [ "$(wc -c < "$MODEL_PATH")" -gt 1000000 ]; then
  echo "$MODEL_PATH already exists"
  exit 0
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.12 is required for the LiteRT/TFLite exporter." >&2
  echo "On macOS: brew install python@3.12" >&2
  exit 1
fi

if [ ! -s "$PT_PATH" ] || [ "$(wc -c < "$PT_PATH")" -le 1000000 ]; then
  rm -f "$PT_PATH"
  echo "Downloading $PT_PATH"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$PT_URL" -o "$PT_PATH"
  elif command -v wget >/dev/null 2>&1; then
    wget "$PT_URL" -O "$PT_PATH"
  else
    echo "Install curl or wget, then rerun this script." >&2
    exit 1
  fi
fi

if [ ! -f .venv-tflite/bin/activate ]; then
  "$PYTHON_BIN" -m venv .venv-tflite
fi

. .venv-tflite/bin/activate

python -m pip install --upgrade pip
python -m pip install ultralytics "litert-torch>=0.9.0" "ai-edge-litert>=2.1.4"

MPLCONFIGDIR="${TMPDIR:-/tmp}" python - <<'PY'
from ultralytics import YOLO

model = YOLO("models/yolov8n-seg.pt")
model.export(format="tflite", imgsz=640)
PY

if [ ! -s "$MODEL_PATH" ] || [ "$(wc -c < "$MODEL_PATH")" -le 1000000 ]; then
  echo "TFLite export did not create a valid $MODEL_PATH" >&2
  exit 1
fi

echo "Ready: $MODEL_PATH"
