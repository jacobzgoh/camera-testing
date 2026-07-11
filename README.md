# Person Tracking Camera

Minimal application that opens a camera, detects people, and draws green tracking boxes around each person.

The preferred runner uses TensorFlow Lite/LiteRT for inference and OpenCV for camera capture, display, non-maximum suppression, and lightweight box tracking. The older C++ ONNX runner is still available as a fallback.

## Run on macOS with TFLite / LiteRT

Install Python 3.12 first:

```sh
brew install python@3.12
```

Export the TFLite model:

```sh
sh scripts/export_tflite_model.sh
```

The script downloads the official Ultralytics `yolov8n-seg.pt` weights, creates `.venv-tflite`, installs the LiteRT exporter/runtime packages, and exports `models/yolov8n-seg.tflite`.

Run:

```sh
. .venv-tflite/bin/activate
python scripts/run_tflite_camera.py --model models/yolov8n-seg.tflite --camera 0
```

For crowded pool footage:

```sh
. .venv-tflite/bin/activate
python scripts/run_tflite_camera.py --model models/yolov8n-seg.tflite --camera 0 --conf 0.20 --nms 0.60 --max-detections 150 --track-iou 0.20 --track-lost 15
```

Press `q` or `Esc` to quit.

Check that LiteRT can load the exported model without opening the camera:

```sh
. .venv-tflite/bin/activate
python scripts/run_tflite_camera.py --model models/yolov8n-seg.tflite --check-model
```

## C++ ONNX Fallback

Install OpenCV first:

```sh
brew install opencv cmake
```

Download the ONNX model:

```sh
sh scripts/download_model.sh
```

Build:

```sh
cmake -S . -B build
cmake --build build
```

Run:

```sh
./build/person_outline --model models/yolov8n-seg.onnx --camera 0
```

## Run Options

```text
--model PATH       YOLO model path
--camera INDEX     Camera index for OpenCV VideoCapture
--width PIXELS     Capture width request
--height PIXELS    Capture height request
--input PIXELS     Square model input size
--conf FLOAT       Person confidence threshold
--nms FLOAT        NMS threshold
--max-detections N Keep up to this many people after NMS
--track-iou FLOAT  Minimum IoU to match a person across frames
--track-lost N     Frames to keep a missing person on screen
--mask FLOAT       Accepted for old commands; boxes do not use masks
--headless         Run without a display window and print FPS only
--check-model      Load the model and exit without opening the camera
```

For busy pool footage, start with the defaults. If the camera misses too many people, lower `--conf` a little:

```sh
python scripts/run_tflite_camera.py --model models/yolov8n-seg.tflite --camera 0 --conf 0.20 --max-detections 150
```

If boxes jump IDs too often, lower `--track-iou` slightly. If boxes linger too long after a person disappears, lower `--track-lost`.

The downloaded models have a fixed `640x640` input. Keep `--input 640` with these models. To use a smaller input for embedded performance, export a separate model at that exact size first; changing only the command-line value is not sufficient.

Reducing camera capture size can still lower display costs without changing the model:

```sh
python scripts/run_tflite_camera.py --model models/yolov8n-seg.tflite --input 640 --width 640 --height 480
```

Check that OpenCV can load the exported model without opening the camera:

```sh
./build/person_outline --model models/yolov8n-seg.onnx --check-model
```

## QNX / Raspberry Pi 5 Notes

The application core is portable C++17 and OpenCV, but the current QNX Raspberry Pi 5 BSP does not list a CSI camera driver. Treat camera integration as required platform work, not as a ready-made OpenCV camera index.

The current TFLite runner is Python-based for Mac testing. For QNX deployment, use the same detection/tracking logic but link against a QNX-compatible TensorFlow Lite or LiteRT C/C++ runtime.

The QNX build needs:

- OpenCV built for QNX with `core`, `imgproc`, `videoio`, `highgui`, and `dnn`
- A camera source exposed to OpenCV `VideoCapture`, or a small replacement for the `cv::VideoCapture` section in `src/main.cpp`
- The TFLite model copied to the target filesystem

QNX SDP 8.0.3 and the Raspberry Pi 5 BSP officially use a Linux or Windows development host. On a Mac, use a supported Linux or Windows VM for the QNX cross-build tools.

The main hardware-specific part is camera acquisition. The Camera Module 3 autofocus control also depends on the eventual QNX sensor/lens driver. Keep the segmentation and drawing code as-is and replace this block:

```cpp
cv::VideoCapture camera(options.camera_index);
camera >> frame;
```

with your QNX camera frame source that returns a BGR `cv::Mat`.

The rest of the pipeline is:

```text
BGR frame
  -> letterbox to model input
  -> YOLO segmentation inference
  -> keep person class only
  -> non-maximum suppression
  -> match boxes to existing track IDs
  -> draw green person boxes
```

## Model License

Ultralytics currently distributes its YOLO models under AGPL-3.0 by default. Review those terms before deployment. A proprietary or commercial embedded product generally requires an Ultralytics commercial license or replacement with a model whose license fits the product.
