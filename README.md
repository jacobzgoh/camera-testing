# Person Tracking Camera

Minimal C++ application that opens a camera, detects people, and draws green tracking boxes around each person.

The app uses OpenCV for camera capture, display, DNN inference, non-maximum suppression, and lightweight box tracking. It expects a YOLO segmentation ONNX model such as `yolov8n-seg.onnx`, but the live display uses boxes instead of segmentation outlines because boxes are more stable in crowded pool footage.

## Build on macOS

Install OpenCV first:

```sh
brew install opencv cmake
```

Download the model:

```sh
sh scripts/download_model.sh
```

The script downloads the official Ultralytics `yolov8n-seg.pt` weights, creates a local `.venv` if needed, and exports `models/yolov8n-seg.onnx`.

Build:

```sh
cmake -S . -B build
cmake --build build
```

Run:

```sh
./build/person_outline --model models/yolov8n-seg.onnx --camera 0
```

Press `q` or `Esc` to quit.

## Run Options

```text
--model PATH       YOLO segmentation ONNX model path
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
./build/person_outline --model models/yolov8n-seg.onnx --camera 0 --conf 0.20 --max-detections 150
```

If boxes jump IDs too often, lower `--track-iou` slightly. If boxes linger too long after a person disappears, lower `--track-lost`.

The downloaded model has a fixed `640x640` input. Keep `--input 640` with this model. To use a smaller input for embedded performance, export a separate model at that exact size first; changing only the command-line value is not sufficient.

Reducing camera capture size can still lower mask-resize and display costs without changing the model:

```sh
./build/person_outline --model models/yolov8n-seg.onnx --input 640 --width 640 --height 480
```

Check that OpenCV can load the exported model without opening the camera:

```sh
./build/person_outline --model models/yolov8n-seg.onnx --check-model
```

## QNX / Raspberry Pi 5 Notes

The application core is portable C++17 and OpenCV, but the current QNX Raspberry Pi 5 BSP does not list a CSI camera driver. Treat camera integration as required platform work, not as a ready-made OpenCV camera index.

The QNX build needs:

- OpenCV built for QNX with `core`, `imgproc`, `videoio`, `highgui`, and `dnn`
- A camera source exposed to OpenCV `VideoCapture`, or a small replacement for the `cv::VideoCapture` section in `src/main.cpp`
- The ONNX model copied to the target filesystem

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
