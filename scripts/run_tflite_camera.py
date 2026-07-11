#!/usr/bin/env python3
import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


PERSON_CLASS_ID = 0
MASK_CHANNELS = 32
DEFAULT_INPUT_SIZE = 640


@dataclass
class Letterbox:
    image: np.ndarray
    scale: float
    pad_x: int
    pad_y: int


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    confidence: float


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]
    confidence: float
    missing_frames: int = 0
    age: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track people with a TFLite/LiteRT YOLO model.")
    parser.add_argument("--model", default="models/yolov8n-seg.tflite", help="TFLite/LiteRT model path")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=1280, help="Capture width request")
    parser.add_argument("--height", type=int, default=720, help="Capture height request")
    parser.add_argument("--input", type=int, default=DEFAULT_INPUT_SIZE, help="Square model input size")
    parser.add_argument("--conf", type=float, default=0.25, help="Person confidence threshold")
    parser.add_argument("--nms", type=float, default=0.50, help="NMS threshold")
    parser.add_argument("--max-detections", type=int, default=100, help="Keep up to this many people after NMS")
    parser.add_argument("--track-iou", type=float, default=0.25, help="Minimum IoU to match a person across frames")
    parser.add_argument("--track-lost", type=int, default=12, help="Frames to keep a missing person on screen")
    parser.add_argument("--headless", action="store_true", help="Run without a display window and print FPS only")
    parser.add_argument("--check-model", action="store_true", help="Load the model and exit without opening the camera")
    args = parser.parse_args()

    if args.input <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not (0.0 <= args.conf <= 1.0 and 0.0 <= args.nms <= 1.0 and 0.0 <= args.track_iou <= 1.0):
        raise ValueError("Thresholds must be between 0 and 1")
    if args.max_detections <= 0 or args.track_lost < 0:
        raise ValueError("--max-detections must be positive and --track-lost must be non-negative")
    return args


def make_letterbox(frame: np.ndarray, size: int) -> Letterbox:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    scaled_width = round(width * scale)
    scaled_height = round(height * scale)
    pad_x = (size - scaled_width) // 2
    pad_y = (size - scaled_height) // 2

    resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    output = np.full((size, size, 3), 114, dtype=frame.dtype)
    output[pad_y : pad_y + scaled_height, pad_x : pad_x + scaled_width] = resized
    return Letterbox(output, scale, pad_x, pad_y)


def prepare_input(letterbox: Letterbox) -> np.ndarray:
    rgb = cv2.cvtColor(letterbox.image, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(chw, axis=0)


def flatten_yolo_output(output: np.ndarray) -> np.ndarray:
    squeezed = np.squeeze(output, axis=0)
    if squeezed.shape[0] <= squeezed.shape[1]:
        return squeezed.T
    return squeezed


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(0, min(width, x + w))
    y2 = max(0, min(height, y + h))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def parse_detections(
    output: np.ndarray,
    letterbox: Letterbox,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> list[Detection]:
    predictions = flatten_yolo_output(output)
    class_count = predictions.shape[1] - 4 - MASK_CHANNELS
    if class_count <= PERSON_CLASS_ID:
        raise RuntimeError("Model output does not contain a person class")

    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    for row in predictions:
        person_score = float(row[4 + PERSON_CLASS_ID])
        if person_score < confidence_threshold:
            continue

        center_x, center_y, box_width, box_height = map(float, row[:4])
        x1 = (center_x - box_width / 2.0 - letterbox.pad_x) / letterbox.scale
        y1 = (center_y - box_height / 2.0 - letterbox.pad_y) / letterbox.scale
        x2 = (center_x + box_width / 2.0 - letterbox.pad_x) / letterbox.scale
        y2 = (center_y + box_height / 2.0 - letterbox.pad_y) / letterbox.scale
        box = clamp_box(
            (round(x1), round(y1), round(x2 - x1), round(y2 - y1)),
            frame_width,
            frame_height,
        )
        if box[2] > 0 and box[3] > 0:
            boxes.append(box)
            scores.append(person_score)

    keep = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, top_k=max_detections)
    if len(keep) == 0:
        return []
    return [Detection(boxes[int(index)], scores[int(index)]) for index in np.array(keep).flatten()]


def intersection_over_union(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if overlap <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - overlap
    return 0.0 if union <= 0.0 else overlap / union


def update_tracks(
    tracks: list[Track],
    detections: list[Detection],
    next_track_id: int,
    iou_threshold: float,
    max_missing_frames: int,
) -> tuple[list[Track], int]:
    matched = [False] * len(detections)

    for track in tracks:
        best_index = -1
        best_iou = iou_threshold
        for index, detection in enumerate(detections):
            if matched[index]:
                continue
            iou = intersection_over_union(track.box, tuple(float(v) for v in detection.box))
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index >= 0:
            detected = tuple(float(v) for v in detections[best_index].box)
            track.box = tuple(track.box[i] * 0.65 + detected[i] * 0.35 for i in range(4))
            track.confidence = detections[best_index].confidence
            track.missing_frames = 0
            track.age += 1
            matched[best_index] = True
        else:
            track.missing_frames += 1
            track.confidence *= 0.85
            track.age += 1

    tracks = [track for track in tracks if track.missing_frames <= max_missing_frames]

    for index, detection in enumerate(detections):
        if not matched[index]:
            tracks.append(Track(next_track_id, tuple(float(v) for v in detection.box), detection.confidence, 0, 1))
            next_track_id += 1

    return tracks, next_track_id


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> None:
    green = (0, 255, 0)
    stale_green = (0, 130, 0)
    frame_height, frame_width = frame.shape[:2]

    for track in tracks:
        x, y, w, h = clamp_box(tuple(round(v) for v in track.box), frame_width, frame_height)
        if w <= 0 or h <= 0:
            continue

        active = track.missing_frames == 0
        color = green if active else stale_green
        thickness = 3 if active else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

        label = f"ID {track.id} {track.confidence * 100:.0f}%"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(0, y - text_height - baseline - 6)
        cv2.rectangle(frame, (x, label_y), (min(frame_width, x + text_width + 10), label_y + text_height + baseline + 6), color, -1)
        cv2.putText(frame, label, (x + 5, label_y + text_height + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    label = f"{fps:.1f} FPS"
    cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        raise RuntimeError("Expected one model input")
    if len(output_details) < 1:
        raise RuntimeError("Expected at least one model output")

    input_shape = tuple(int(v) for v in input_details[0]["shape"])
    if input_shape != (1, 3, args.input, args.input):
        raise RuntimeError(f"Expected model input shape (1, 3, {args.input}, {args.input}), got {input_shape}")

    if args.check_model:
        print(
            f"Model check passed: {args.model} "
            f"(input {input_shape}, outputs {[tuple(int(v) for v in detail['shape']) for detail in output_details]})"
        )
        return 0

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not args.headless:
        cv2.namedWindow("Person Tracking TFLite", cv2.WINDOW_NORMAL)

    tracks: list[Track] = []
    next_track_id = 1
    fps = 0.0
    last_time = time.monotonic()

    while True:
        ok, frame = camera.read()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("Camera returned an empty frame")

        letterbox = make_letterbox(frame, args.input)
        interpreter.set_tensor(input_details[0]["index"], prepare_input(letterbox))
        interpreter.invoke()
        outputs = [interpreter.get_tensor(detail["index"]) for detail in output_details]

        detections = parse_detections(
            outputs[0],
            letterbox,
            frame.shape[1],
            frame.shape[0],
            args.conf,
            args.nms,
            args.max_detections,
        )
        tracks, next_track_id = update_tracks(tracks, detections, next_track_id, args.track_iou, args.track_lost)
        draw_tracks(frame, tracks)

        now = time.monotonic()
        seconds = now - last_time
        last_time = now
        if seconds > 0.0:
            instant_fps = 1.0 / seconds
            fps = instant_fps if fps == 0.0 else fps * 0.9 + instant_fps * 0.1

        if args.headless:
            print(f"\r{fps:.1f} FPS, detections: {len(detections)}, tracks: {len(tracks)}", end="", flush=True)
        else:
            draw_fps(frame, fps)
            cv2.imshow("Person Tracking TFLite", frame)
            key = cv2.waitKey(1)
            if key in (27, ord("q"), ord("Q")):
                break

    if args.headless:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
