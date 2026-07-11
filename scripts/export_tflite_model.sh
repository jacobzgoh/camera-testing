#!/usr/bin/env sh
set -eu

mkdir -p models

MODEL_NAME="${1:-yolov8n}"
IMG_SIZE="${2:-640}"
PT_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/${MODEL_NAME}.pt"
PT_PATH="models/${MODEL_NAME}.pt"
MODEL_PATH="models/${MODEL_NAME}-${IMG_SIZE}.tflite"
EXPORT_PATH="models/${MODEL_NAME}.tflite"
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

if ! python - <<'PY' >/dev/null 2>&1
import ai_edge_litert
import litert_torch
import ultralytics
PY
then
  python -m pip install --upgrade pip
  python -m pip install ultralytics "litert-torch>=0.9.0" "ai-edge-litert>=2.1.4"
fi

MODEL_NAME="$MODEL_NAME" IMG_SIZE="$IMG_SIZE" MPLCONFIGDIR="${TMPDIR:-/tmp}" python - <<'PY'
import os
from ultralytics import YOLO

model_name = os.environ["MODEL_NAME"]
img_size = int(os.environ["IMG_SIZE"])

model = YOLO(f"models/{model_name}.pt")
model.export(format="tflite", imgsz=img_size)
PY

if [ -s "$EXPORT_PATH" ]; then
  mv "$EXPORT_PATH" "$MODEL_PATH"
fi

if [ ! -s "$MODEL_PATH" ] || [ "$(wc -c < "$MODEL_PATH")" -le 1000000 ]; then
  echo "TFLite export did not create a valid $MODEL_PATH" >&2
  exit 1
fi

echo "Ready: $MODEL_PATH"
