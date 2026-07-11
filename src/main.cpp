#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

constexpr int kPersonClassId = 0;
constexpr int kMaskChannels = 32;
constexpr int kDefaultInputSize = 640;

struct Options {
    std::string model_path = "models/yolov8n-seg.onnx";
    int camera_index = 0;
    int capture_width = 1280;
    int capture_height = 720;
    int input_size = kDefaultInputSize;
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.50f;
    float mask_threshold = 0.50f;
    float track_iou_threshold = 0.25f;
    int max_detections = 100;
    int max_missing_frames = 12;
    bool headless = false;
    bool check_model = false;
};

struct Letterbox {
    cv::Mat image;
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

struct Detection {
    cv::Rect box;
    float confidence = 0.0f;
};

struct Track {
    int id = 0;
    cv::Rect2f box;
    float confidence = 0.0f;
    int missing_frames = 0;
    int age = 0;
};

struct ProtoView {
    cv::Mat data;
    int height = 0;
    int width = 0;
};

struct ModelOutputs {
    const cv::Mat* detections = nullptr;
    const cv::Mat* prototypes = nullptr;
};

void print_usage(const char* binary_name) {
    std::cout
        << "Usage: " << binary_name << " [options]\n\n"
        << "Options:\n"
        << "  --model PATH       YOLO segmentation ONNX model path (default: models/yolov8n-seg.onnx)\n"
        << "  --camera INDEX     Camera index for OpenCV VideoCapture (default: 0)\n"
        << "  --width PIXELS     Capture width request (default: 1280)\n"
        << "  --height PIXELS    Capture height request (default: 720)\n"
        << "  --input PIXELS     Square model input size (default: 640)\n"
        << "  --conf FLOAT       Person confidence threshold (default: 0.25)\n"
        << "  --nms FLOAT        NMS threshold (default: 0.50)\n"
        << "  --max-detections N Keep up to this many people after NMS (default: 100)\n"
        << "  --track-iou FLOAT  Minimum IoU to match a person across frames (default: 0.25)\n"
        << "  --track-lost N     Frames to keep a missing person on screen (default: 12)\n"
        << "  --mask FLOAT       Accepted for old commands; boxes do not use masks\n"
        << "  --headless         Run without a display window and print FPS only\n"
        << "  --check-model      Load the model and exit without opening the camera\n"
        << "  --help             Show this message\n";
}

template <typename T>
T parse_value(const std::string& raw, const std::string& name) {
    std::istringstream stream(raw);
    T value{};
    stream >> value;
    if (!stream || !stream.eof()) {
        throw std::runtime_error("Invalid value for " + name + ": " + raw);
    }
    return value;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model") {
            options.model_path = require_value(arg);
        } else if (arg == "--camera") {
            options.camera_index = parse_value<int>(require_value(arg), arg);
        } else if (arg == "--width") {
            options.capture_width = parse_value<int>(require_value(arg), arg);
        } else if (arg == "--height") {
            options.capture_height = parse_value<int>(require_value(arg), arg);
        } else if (arg == "--input") {
            options.input_size = parse_value<int>(require_value(arg), arg);
        } else if (arg == "--conf") {
            options.confidence_threshold = parse_value<float>(require_value(arg), arg);
        } else if (arg == "--nms") {
            options.nms_threshold = parse_value<float>(require_value(arg), arg);
        } else if (arg == "--mask") {
            options.mask_threshold = parse_value<float>(require_value(arg), arg);
        } else if (arg == "--max-detections") {
            options.max_detections = parse_value<int>(require_value(arg), arg);
        } else if (arg == "--track-iou") {
            options.track_iou_threshold = parse_value<float>(require_value(arg), arg);
        } else if (arg == "--track-lost") {
            options.max_missing_frames = parse_value<int>(require_value(arg), arg);
        } else if (arg == "--headless") {
            options.headless = true;
        } else if (arg == "--check-model") {
            options.check_model = true;
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.input_size <= 0 || options.capture_width <= 0 || options.capture_height <= 0) {
        throw std::runtime_error("Image dimensions must be positive");
    }
    if (options.confidence_threshold < 0.0f || options.confidence_threshold > 1.0f ||
        options.nms_threshold < 0.0f || options.nms_threshold > 1.0f ||
        options.mask_threshold < 0.0f || options.mask_threshold > 1.0f ||
        options.track_iou_threshold < 0.0f || options.track_iou_threshold > 1.0f) {
        throw std::runtime_error("Thresholds must be between 0 and 1");
    }
    if (options.max_detections <= 0 || options.max_missing_frames < 0) {
        throw std::runtime_error("--max-detections must be positive and --track-lost must be non-negative");
    }

    return options;
}

ModelOutputs identify_outputs(const std::vector<cv::Mat>& outputs) {
    ModelOutputs result;
    for (const cv::Mat& output : outputs) {
        if (output.dims == 3 && result.detections == nullptr) {
            result.detections = &output;
        } else if (output.dims == 4 && result.prototypes == nullptr) {
            result.prototypes = &output;
        }
    }
    if (result.detections == nullptr || result.prototypes == nullptr) {
        throw std::runtime_error(
            "Model is not a supported YOLOv8 segmentation export: expected one 3D detection output and one 4D mask output");
    }
    return result;
}

std::vector<cv::Mat> run_model(cv::dnn::Net& net, const cv::Mat& image, int input_size) {
    cv::Mat blob = cv::dnn::blobFromImage(
        image,
        1.0 / 255.0,
        cv::Size(input_size, input_size),
        cv::Scalar(),
        true,
        false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    try {
        net.forward(outputs, net.getUnconnectedOutLayersNames());
    } catch (const cv::Exception& error) {
        throw std::runtime_error(
            "Model inference failed at --input " + std::to_string(input_size) +
            ". The bundled model is exported for 640x640; use --input 640 or export a model at the requested size. OpenCV: " +
            std::string(error.what()));
    }
    identify_outputs(outputs);
    return outputs;
}

Letterbox make_letterbox(const cv::Mat& frame, int size) {
    const float scale = std::min(size / static_cast<float>(frame.cols), size / static_cast<float>(frame.rows));
    const int scaled_width = static_cast<int>(std::round(frame.cols * scale));
    const int scaled_height = static_cast<int>(std::round(frame.rows * scale));
    const int pad_x = (size - scaled_width) / 2;
    const int pad_y = (size - scaled_height) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(scaled_width, scaled_height), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat output(size, size, frame.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(output(cv::Rect(pad_x, pad_y, resized.cols, resized.rows)));

    return {output, scale, pad_x, pad_y};
}

cv::Mat flatten_output(const cv::Mat& output) {
    if (output.dims != 3) {
        throw std::runtime_error("Expected YOLO detection output with 3 dimensions");
    }

    const int rows = output.size[1];
    const int cols = output.size[2];
    cv::Mat shaped(rows, cols, CV_32F, const_cast<float*>(reinterpret_cast<const float*>(output.data)));

    if (cols <= rows) {
        return shaped.clone();
    }

    cv::Mat transposed;
    cv::transpose(shaped, transposed);
    return transposed;
}

ProtoView flatten_proto(const cv::Mat& proto) {
    if (proto.dims != 4) {
        throw std::runtime_error("Expected YOLO proto output with 4 dimensions");
    }

    const int channels = proto.size[1];
    const int height = proto.size[2];
    const int width = proto.size[3];
    if (channels != kMaskChannels || height <= 0 || width <= 0) {
        throw std::runtime_error(
            "Unsupported mask prototype shape; this application expects 32-channel YOLOv8 segmentation masks");
    }
    return {
        cv::Mat(channels, height * width, CV_32F, const_cast<float*>(reinterpret_cast<const float*>(proto.data))).clone(),
        height,
        width};
}

cv::Rect clamp_rect(const cv::Rect& rect, const cv::Size& bounds) {
    return rect & cv::Rect(0, 0, bounds.width, bounds.height);
}

cv::Rect to_integer_rect(const cv::Rect2f& rect) {
    return cv::Rect(
        cv::Point(static_cast<int>(std::round(rect.x)), static_cast<int>(std::round(rect.y))),
        cv::Point(
            static_cast<int>(std::round(rect.x + rect.width)),
            static_cast<int>(std::round(rect.y + rect.height))));
}

std::vector<Detection> parse_detections(
    const cv::Mat& output,
    const Letterbox& letterbox,
    const cv::Size& frame_size,
    float confidence_threshold,
    float nms_threshold,
    int max_detections) {
    const cv::Mat predictions = flatten_output(output);
    const int attributes = predictions.cols;
    const int class_count = attributes - 4 - kMaskChannels;
    if (class_count <= kPersonClassId) {
        throw std::runtime_error("Model output does not contain a person class");
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (int i = 0; i < predictions.rows; ++i) {
        const float* row = predictions.ptr<float>(i);
        const float person_score = row[4 + kPersonClassId];
        if (person_score < confidence_threshold) {
            continue;
        }

        const float center_x = row[0];
        const float center_y = row[1];
        const float width = row[2];
        const float height = row[3];

        const float x1 = (center_x - width / 2.0f - letterbox.pad_x) / letterbox.scale;
        const float y1 = (center_y - height / 2.0f - letterbox.pad_y) / letterbox.scale;
        const float x2 = (center_x + width / 2.0f - letterbox.pad_x) / letterbox.scale;
        const float y2 = (center_y + height / 2.0f - letterbox.pad_y) / letterbox.scale;

        const cv::Rect unclamped(
            cv::Point(static_cast<int>(std::round(x1)), static_cast<int>(std::round(y1))),
            cv::Point(static_cast<int>(std::round(x2)), static_cast<int>(std::round(y2))));
        const cv::Rect box = clamp_rect(unclamped, frame_size);
        if (box.empty()) {
            continue;
        }

        boxes.push_back(box);
        scores.push_back(person_score);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, keep, 1.0f, max_detections);

    std::vector<Detection> detections;
    detections.reserve(keep.size());
    for (const int index : keep) {
        detections.push_back({boxes[index], scores[index]});
    }
    return detections;
}

float intersection_over_union(const cv::Rect2f& first, const cv::Rect2f& second) {
    const cv::Rect2f overlap = first & second;
    const float overlap_area = overlap.area();
    if (overlap_area <= 0.0f) {
        return 0.0f;
    }

    const float union_area = first.area() + second.area() - overlap_area;
    return union_area <= 0.0f ? 0.0f : overlap_area / union_area;
}

std::vector<Track> update_tracks(
    const std::vector<Track>& current_tracks,
    const std::vector<Detection>& detections,
    int& next_track_id,
    float iou_threshold,
    int max_missing_frames) {
    std::vector<Track> updated = current_tracks;
    std::vector<bool> matched_detection(detections.size(), false);

    for (Track& track : updated) {
        int best_index = -1;
        float best_iou = iou_threshold;
        for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
            if (matched_detection[i]) {
                continue;
            }

            const float iou = intersection_over_union(track.box, cv::Rect2f(detections[i].box));
            if (iou > best_iou) {
                best_iou = iou;
                best_index = i;
            }
        }

        if (best_index >= 0) {
            const cv::Rect2f detected_box(detections[best_index].box);
            track.box.x = track.box.x * 0.65f + detected_box.x * 0.35f;
            track.box.y = track.box.y * 0.65f + detected_box.y * 0.35f;
            track.box.width = track.box.width * 0.65f + detected_box.width * 0.35f;
            track.box.height = track.box.height * 0.65f + detected_box.height * 0.35f;
            track.confidence = detections[best_index].confidence;
            track.missing_frames = 0;
            track.age += 1;
            matched_detection[best_index] = true;
        } else {
            track.missing_frames += 1;
            track.confidence *= 0.85f;
            track.age += 1;
        }
    }

    updated.erase(
        std::remove_if(
            updated.begin(),
            updated.end(),
            [max_missing_frames](const Track& track) {
                return track.missing_frames > max_missing_frames;
            }),
        updated.end());

    for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
        if (!matched_detection[i]) {
            updated.push_back({
                next_track_id++,
                cv::Rect2f(detections[i].box),
                detections[i].confidence,
                0,
                1});
        }
    }

    return updated;
}

void draw_tracks(
    cv::Mat& frame,
    const std::vector<Track>& tracks) {
    const cv::Scalar green(0, 255, 0);
    const cv::Scalar stale_green(0, 130, 0);
    for (const Track& track : tracks) {
        const cv::Rect box = clamp_rect(to_integer_rect(track.box), frame.size());
        if (box.empty()) {
            continue;
        }

        const bool active = track.missing_frames == 0;
        const cv::Scalar color = active ? green : stale_green;
        const int thickness = active ? 3 : 2;
        cv::rectangle(frame, box, color, thickness, cv::LINE_AA);

        std::ostringstream label;
        label.precision(0);
        label << "ID " << track.id << " " << std::fixed << track.confidence * 100.0f << "%";
        const std::string text = label.str();

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
        const int label_y = std::max(0, box.y - text_size.height - baseline - 6);
        const cv::Rect label_background(
            box.x,
            label_y,
            std::min(text_size.width + 10, frame.cols - box.x),
            text_size.height + baseline + 6);
        cv::rectangle(frame, label_background, color, cv::FILLED);
        cv::putText(
            frame,
            text,
            cv::Point(box.x + 5, label_y + text_size.height + 1),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0, 0, 0),
            2,
            cv::LINE_AA);
    }
}

void draw_fps(cv::Mat& frame, double fps) {
    std::ostringstream label;
    label.precision(1);
    label << std::fixed << fps << " FPS";
    cv::putText(frame, label.str(), cv::Point(16, 36), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(frame, label.str(), cv::Point(16, 36), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        cv::dnn::Net net = cv::dnn::readNetFromONNX(options.model_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        if (options.check_model) {
            const cv::Mat test_image(options.input_size, options.input_size, CV_8UC3, cv::Scalar(114, 114, 114));
            const std::vector<cv::Mat> outputs = run_model(net, test_image, options.input_size);
            const ModelOutputs model_outputs = identify_outputs(outputs);
            const ProtoView proto = flatten_proto(*model_outputs.prototypes);
            const cv::Mat detections = flatten_output(*model_outputs.detections);
            std::cout << "Model check passed: " << options.model_path
                      << " (input " << options.input_size << "x" << options.input_size
                      << ", predictions " << detections.rows
                      << ", mask prototypes " << proto.data.rows << "x" << proto.height << "x" << proto.width
                      << ")\n";
            return 0;
        }

        cv::VideoCapture camera(options.camera_index);
        if (!camera.isOpened()) {
            throw std::runtime_error("Could not open camera index " + std::to_string(options.camera_index));
        }
        camera.set(cv::CAP_PROP_FRAME_WIDTH, options.capture_width);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, options.capture_height);

        if (!options.headless) {
            cv::namedWindow("Person Tracking", cv::WINDOW_NORMAL);
        }

        auto last_time = std::chrono::steady_clock::now();
        double fps = 0.0;
        std::vector<Track> tracks;
        int next_track_id = 1;

        while (true) {
            cv::Mat frame;
            camera >> frame;
            if (frame.empty()) {
                throw std::runtime_error("Camera returned an empty frame");
            }

            const Letterbox letterbox = make_letterbox(frame, options.input_size);
            const std::vector<cv::Mat> outputs = run_model(net, letterbox.image, options.input_size);
            const ModelOutputs model_outputs = identify_outputs(outputs);
            const std::vector<Detection> detections = parse_detections(
                *model_outputs.detections,
                letterbox,
                frame.size(),
                options.confidence_threshold,
                options.nms_threshold,
                options.max_detections);

            tracks = update_tracks(
                tracks,
                detections,
                next_track_id,
                options.track_iou_threshold,
                options.max_missing_frames);
            draw_tracks(frame, tracks);

            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now - last_time).count();
            last_time = now;
            if (seconds > 0.0) {
                fps = fps == 0.0 ? 1.0 / seconds : fps * 0.9 + (1.0 / seconds) * 0.1;
            }

            if (options.headless) {
                std::cout << "\r" << fps << " FPS, detections: " << detections.size()
                          << ", tracks: " << tracks.size() << std::flush;
            } else {
                draw_fps(frame, fps);
                cv::imshow("Person Tracking", frame);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') {
                    break;
                }
            }
        }

        if (options.headless) {
            std::cout << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "person_outline: " << error.what() << "\n";
        return 1;
    }
}
