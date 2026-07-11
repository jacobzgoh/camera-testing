# Person Tracking Camera

Minimal application that opens a camera, detects people, and draws green tracking boxes around each person.

The preferred runner for the best FPS/tracking balance is the C++ ONNX app. It uses OpenCV for camera capture, DNN inference, display, non-maximum suppression, and motion-aware box tracking. The TensorFlow Lite/LiteRT runner is still available for comparison.

## Recommended: C++ ONNX

Install OpenCV first:

```sh
brew install opencv cmake
```

Export the balanced box-only ONNX model:

```sh
sh scripts/download_model.sh yolov8n 640
```

Build:

```sh
cmake -S . -B build
cmake --build build
```

Run the balanced model:

```sh
./build/person_outline --model models/yolov8n-640.onnx --input 640 --camera 0 --conf 0.20 --nms 0.60 --max-detections 150 --track-iou 0.20 --track-lost 15
```

For higher FPS, export and run the smaller 416 model:

```sh
sh scripts/download_model.sh yolov8n 416
./build/person_outline --model models/yolov8n-416.onnx --input 416 --camera 0 --conf 0.20 --nms 0.60 --max-detections 150 --track-iou 0.20 --track-lost 15
```

For the old segmentation baseline:

```sh
sh scripts/download_model.sh yolov8n-seg 640
./build/person_outline --model models/yolov8n-seg-640.onnx --input 640 --camera 0 --conf 0.20 --nms 0.60 --max-detections 150 --track-iou 0.20 --track-lost 15
```

Press `q` or `Esc` to quit.

Check that OpenCV can load a model without opening the camera:

```sh
./build/person_outline --model models/yolov8n-640.onnx --input 640 --check-model
```

## TensorFlow Lite / LiteRT Comparison

Install Python 3.12 first:

```sh
brew install python@3.12
```

Export and run a TFLite model:

```sh
sh scripts/export_tflite_model.sh yolov8n 640
. .venv-tflite/bin/activate
python3 scripts/run_tflite_camera.py --model models/yolov8n-640.tflite --camera 0
```

The TFLite path is useful if the team wants that runtime, but on this Mac the C++ ONNX path is the recommended default for FPS.

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
--qnx-camera       Use QNX Camera API instead of OpenCV VideoCapture
--opencv-camera    Use OpenCV VideoCapture instead of QNX Camera API
--mask FLOAT       Accepted for old commands; boxes do not use masks
--headless         Run without a display window and print FPS only
--check-model      Load the model and exit without opening the camera
```

For busy pool footage, start with the defaults. If the camera misses too many people, lower `--conf` a little:

```sh
./build/person_outline --model models/yolov8n-640.onnx --input 640 --camera 0 --conf 0.20 --max-detections 150
```

If boxes jump IDs too often, lower `--track-iou` slightly. If boxes linger too long after a person disappears, lower `--track-lost`.

The downloaded models have a fixed `640x640` input. Keep `--input 640` with these models. To use a smaller input for embedded performance, export a separate model at that exact size first; changing only the command-line value is not sufficient.

Reducing camera capture size can still lower display costs without changing the model:

```sh
./build/person_outline --model models/yolov8n-640.onnx --input 640 --width 640 --height 480
```

Check that OpenCV can load the exported model without opening the camera:

```sh
./build/person_outline --model models/yolov8n-640.onnx --input 640 --check-model
```

## QNX / Raspberry Pi 5 Notes

On QNX, the app uses the native QNX Camera API by default instead of OpenCV `VideoCapture`. The camera path follows the QNX AI camera sample pattern:

```text
camera_get_supported_cameras
  -> camera_open
  -> camera_set_buffer_retrieval_mode(CAMERA_BRM_LATEST_FLUSH)
  -> camera_enable_viewfinder_event / camera_enable_status_event
  -> camera_start_viewfinder
  -> MsgReceivePulse
  -> camera_get_viewfinder_buffers
  -> convert QNX frame buffer to BGR cv::Mat
  -> camera_return_buffer
```

Supported camera frame conversions currently include:

- `CAMERA_FRAMETYPE_YCBYCR` -> `cv::COLOR_YUV2BGR_YUY2`
- `CAMERA_FRAMETYPE_CBYCRY` -> `cv::COLOR_YUV2BGR_UYVY`
- `CAMERA_FRAMETYPE_YCRYCB` -> `cv::COLOR_YUV2BGR_YVYU`
- `CAMERA_FRAMETYPE_BGR8888` -> `cv::COLOR_BGRA2BGR`
- `CAMERA_FRAMETYPE_RGB8888` -> `cv::COLOR_RGBA2BGR`
- `CAMERA_FRAMETYPE_RGB888` -> `cv::COLOR_RGB2BGR`
- `CAMERA_FRAMETYPE_GRAY8` -> `cv::COLOR_GRAY2BGR`

The application core is portable C++17 and OpenCV. The hardware-specific part is whether the QNX Raspberry Pi 5 image exposes the Camera Module 3 through the QNX Camera API. If it does, the app should use that camera path directly. If it only exposes a generic OpenCV-compatible source, pass `--opencv-camera`.

The QNX build needs:

- OpenCV built for QNX with `core`, `imgproc`, `videoio`, `highgui`, and `dnn`
- QNX Camera API headers and library
- A camera exposed through the QNX Camera API, or an OpenCV-compatible fallback source
- The ONNX model copied to the target filesystem

QNX SDP 8.0.3 and the Raspberry Pi 5 BSP officially use a Linux or Windows development host. On a Mac, use a supported Linux or Windows VM for the QNX cross-build tools.

Example QNX run:

```sh
./person_outline --model yolov8n-640.onnx --input 640 --camera 0 --conf 0.20 --nms 0.60 --max-detections 150 --track-iou 0.20 --track-lost 15
```

Force OpenCV camera capture on QNX:

```sh
./person_outline --opencv-camera --model yolov8n-640.onnx --input 640 --camera 0
```

The Camera Module 3 autofocus control depends on the QNX sensor/lens driver. The app consumes frames once the camera stack provides them.

The rest of the pipeline is:

```text
BGR frame
  -> letterbox to model input
  -> YOLO detection inference
  -> keep person class only
  -> non-maximum suppression
  -> match boxes to existing track IDs with motion prediction
  -> draw green person boxes
```

## Model License

Ultralytics currently distributes its YOLO models under AGPL-3.0 by default. Review those terms before deployment. A proprietary or commercial embedded product generally requires an Ultralytics commercial license or replacement with a model whose license fits the product.
