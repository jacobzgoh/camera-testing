#!/usr/bin/env sh
set -eu

mkdir -p models

MODEL_NAME="${1:-yolov8n}"
IMG_SIZE="${2:-640}"
PT_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/${MODEL_NAME}.pt"
PT_PATH="models/${MODEL_NAME}.pt"
MODEL_PATH="models/${MODEL_NAME}-${IMG_SIZE}.onnx"
EXPORT_PATH="models/${MODEL_NAME}.onnx"

if [ -s "$MODEL_PATH" ] && [ "$(wc -c < "$MODEL_PATH")" -gt 1000000 ]; then
  echo "$MODEL_PATH already exists"
  exit 0
fi

if [ -f "$MODEL_PATH" ]; then
  echo "Removing invalid or incomplete $MODEL_PATH"
  rm -f "$MODEL_PATH"
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

if ! command -v python3 >/dev/null 2>&1; then
  echo "Downloaded $PT_PATH, but python3 is required to export ONNX." >&2
  exit 1
fi

if [ ! -f .venv/bin/activate ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate

MODEL_NAME="$MODEL_NAME" IMG_SIZE="$IMG_SIZE" python - <<'PY'
import importlib.util
import os
import subprocess
import sys

if importlib.util.find_spec("ultralytics") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

from ultralytics import YOLO

model_name = os.environ["MODEL_NAME"]
img_size = int(os.environ["IMG_SIZE"])

model = YOLO(f"models/{model_name}.pt")
model.export(format="onnx", imgsz=img_size, opset=12, simplify=False)
PY

if [ -s "$EXPORT_PATH" ]; then
  mv "$EXPORT_PATH" "$MODEL_PATH"
fi

if command -v curl >/dev/null 2>&1; then
  true
fi

if [ ! -s "$MODEL_PATH" ] || [ "$(wc -c < "$MODEL_PATH")" -le 1000000 ]; then
  echo "ONNX export did not create a valid $MODEL_PATH" >&2
  exit 1
fi

echo "Ready: $MODEL_PATH"
