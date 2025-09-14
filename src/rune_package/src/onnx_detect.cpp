#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

// ======================= 配置参数 =======================
const std::string ONNX_MODEL_PATH = "/mnt/d/rm能量机关/能量机关数据集/训练/run/exp/weights/best.onnx";
const std::string VIDEO_PATH = "/mnt/d/rm能量机关/能量机关视频素材（黑暗环境）/能量机关视频素材（黑暗环境）/新能量机关_正在激活.mp4";
const std::string OUTPUT_PATH = "/home/icebear/ros2_ws/output_video1.mp4";

const int INPUT_SIZE = 640;
const float CONF_THRESHOLD = 0.4f;
const float IOU_THRESHOLD = 0.5f;

const std::vector<std::string> CLASS_NAMES = {"class0", "class1", "class2"};
const std::vector<cv::Scalar> COLORS = {cv::Scalar(255, 128, 0), cv::Scalar(0, 200, 255), cv::Scalar(255, 0, 120)};

// ======================= 辅助结构体 =======================
struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

// ======================= ROS2 节点类 =======================
class OnnxDetectNode : public rclcpp::Node {
public:
    OnnxDetectNode() : Node("onnx_detect_node"), initialized_(false) {
        initialize();
    }

private:
    // 初始化所有资源
    void initialize() {
        RCLCPP_INFO(this->get_logger(), "Initializing ONNX Detection Node...");

        try {
            // 1. 检查文件路径
            if (!std::filesystem::exists(ONNX_MODEL_PATH)) {
                RCLCPP_FATAL(this->get_logger(), "ONNX model file not found at: %s", ONNX_MODEL_PATH.c_str());
                return;
            }
            if (!std::filesystem::exists(VIDEO_PATH)) {
                RCLCPP_FATAL(this->get_logger(), "Input video file not found at: %s", VIDEO_PATH.c_str());
                return;
            }

            // 2. 初始化 ONNX Runtime
            env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_detect");
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            session_ = Ort::Session(env_, ONNX_MODEL_PATH.c_str(), session_options);
            
            // 获取输入输出名称
            input_name_ = session_.GetInputNameAllocated(0, allocator_).get();
            output_name_ = session_.GetOutputNameAllocated(0, allocator_).get();
            RCLCPP_INFO(this->get_logger(), "Model Loaded. Input: '%s', Output: '%s'", input_name_.c_str(), output_name_.c_str());

            // 3. 初始化视频读写
            cap_.open(VIDEO_PATH);
            if (!cap_.isOpened()) {
                RCLCPP_FATAL(this->get_logger(), "Cannot open video file: %s", VIDEO_PATH.c_str());
                return;
            }
            int width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
            int height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
            double fps = cap_.get(cv::CAP_PROP_FPS);
            if (fps <= 0) fps = 25.0;

            out_.open(OUTPUT_PATH, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
            if (!out_.isOpened()) {
                RCLCPP_FATAL(this->get_logger(), "Cannot create output video file: %s", OUTPUT_PATH.c_str());
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Video I/O initialized. Output will be saved to: %s", OUTPUT_PATH.c_str());

            // 4. 创建定时器开始处理
            initialized_ = true;
            timer_ = this->create_wall_timer(std::chrono::milliseconds(1), std::bind(&OnnxDetectNode::process_frame, this));
            RCLCPP_INFO(this->get_logger(), "Initialization successful. Starting video processing...");

        } catch (const Ort::Exception& e) {
            RCLCPP_FATAL(this->get_logger(), "ONNX Runtime exception: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Standard exception: %s", e.what());
        }
    }

    // 预处理函数 (Letterbox) - 关键修正
    void preprocess(const cv::Mat& frame, std::vector<float>& blob, float& r, int& pad_w, int& pad_h) {
        int h0 = frame.rows;
        int w0 = frame.cols;
        r = std::min(static_cast<float>(INPUT_SIZE) / h0, static_cast<float>(INPUT_SIZE) / w0);
        int nw = static_cast<int>(round(w0 * r));
        int nh = static_cast<int>(round(h0 * r));

        cv::Mat img_resized;
        cv::resize(frame, img_resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

        cv::Mat canvas(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(128, 128, 128));
        pad_w = (INPUT_SIZE - nw) / 2;
        pad_h = (INPUT_SIZE - nh) / 2;
        img_resized.copyTo(canvas(cv::Rect(pad_w, pad_h, nw, nh)));

        cv::Mat rgb;
        cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

        blob.resize(1 * 3 * INPUT_SIZE * INPUT_SIZE);
        float* blob_ptr = blob.data();
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < INPUT_SIZE; ++h) {
                for (int w = 0; w < INPUT_SIZE; ++w) {
                    blob_ptr[c * INPUT_SIZE * INPUT_SIZE + h * INPUT_SIZE + w] = rgb.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }

    // 后处理函数：兼容 [1,A,N] / [1,N,A] / [N,A]，兼容 xywh(+obj)+cls
    std::vector<Detection> postprocess(const float* output, const std::vector<int64_t>& shape,
                                       float r, int pad_w, int pad_h, const cv::Size& orig_shape) {
        const int nc = static_cast<int>(CLASS_NAMES.size());

        // 解析输出形状
        int N = 0;                 // 检测框数量
        int A = 0;                 // 每个框的属性数（4(+obj)+nc）
        enum Layout { NA_2D, N_A_3D, A_N_3D } layout;

        if (shape.size() == 2) {            // [N, A]
            N = static_cast<int>(shape[0]);
            A = static_cast<int>(shape[1]);
            layout = NA_2D;
        } else if (shape.size() == 3) {     // [1, ?, ?]
            int d1 = static_cast<int>(shape[1]);
            int d2 = static_cast<int>(shape[2]);
            // 猜测哪个是 A
            if (d1 == 4 + nc || d1 == 5 + nc) {
                A = d1; N = d2; layout = A_N_3D;   // [1, A, N]
            } else if (d2 == 4 + nc || d2 == 5 + nc) {
                N = d1; A = d2; layout = N_A_3D;   // [1, N, A]
            } else {
                // 回退：按 [1, N, A]
                N = d1; A = d2; layout = N_A_3D;
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unexpected output rank: %zu", shape.size());
            return {};
        }

        const bool has_obj = (A == 5 + nc);

        // 安全访问函数：返回第 i 个框的第 j 个属性
        auto at = [&](int i, int j) -> float {
            if (layout == NA_2D) {
                return output[i * A + j];
            } else if (layout == N_A_3D) { // [1, N, A]
                return output[i * A + j];
            } else {                      // [1, A, N]
                return output[j * N + i];
            }
        };

        // 收集反 letterbox 并裁剪后的框与分数
        std::vector<cv::Rect> boxes_int;
        std::vector<float> scores;
        std::vector<int> class_ids;

        boxes_int.reserve(N);
        scores.reserve(N);
        class_ids.reserve(N);

        for (int i = 0; i < N; ++i) {
            // xywh
            float cx = at(i, 0);
            float cy = at(i, 1);
            float w  = std::max(0.0f, at(i, 2));
            float h  = std::max(0.0f, at(i, 3));

            // 类别分数（取最大类）
            int cls_best = 0;
            float cls_best_score = -1.0f;
            int cls_base = has_obj ? 5 : 4;
            for (int k = 0; k < nc; ++k) {
                float sc = at(i, cls_base + k);
                if (sc > cls_best_score) {
                    cls_best_score = sc;
                    cls_best = k;
                }
            }
            float obj = has_obj ? std::max(0.0f, at(i, 4)) : 1.0f;
            float final_score = obj * cls_best_score;
            if (final_score < CONF_THRESHOLD) continue;

            // xywh -> xyxy (letterbox 坐标)
            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            // 反 letterbox：去 pad，再除以 r
            x1 = (x1 - static_cast<float>(pad_w)) / std::max(r, 1e-6f);
            x2 = (x2 - static_cast<float>(pad_w)) / std::max(r, 1e-6f);
            y1 = (y1 - static_cast<float>(pad_h)) / std::max(r, 1e-6f);
            y2 = (y2 - static_cast<float>(pad_h)) / std::max(r, 1e-6f);

            // 裁剪到原图尺寸
            x1 = std::min(std::max(0.0f, x1), static_cast<float>(orig_shape.width  - 1));
            y1 = std::min(std::max(0.0f, y1), static_cast<float>(orig_shape.height - 1));
            x2 = std::min(std::max(0.0f, x2), static_cast<float>(orig_shape.width  - 1));
            y2 = std::min(std::max(0.0f, y2), static_cast<float>(orig_shape.height - 1));

            int ix = static_cast<int>(std::round(x1));
            int iy = static_cast<int>(std::round(y1));
            int iw = static_cast<int>(std::round(x2 - x1));
            int ih = static_cast<int>(std::round(y2 - y1));
            if (iw <= 0 || ih <= 0) continue;

            boxes_int.emplace_back(ix, iy, iw, ih);
            scores.push_back(final_score);
            class_ids.push_back(cls_best);
        }

        if (boxes_int.empty()) return {};

        // 按类别执行 NMS
        std::vector<Detection> final_dets;
        for (int c = 0; c < nc; ++c) {
            std::vector<cv::Rect> cls_boxes;
            std::vector<float> cls_scores;
            std::vector<int> mapping;
            cls_boxes.reserve(boxes_int.size());
            cls_scores.reserve(scores.size());
            mapping.reserve(boxes_int.size());

            for (size_t i = 0; i < class_ids.size(); ++i) {
                if (class_ids[i] == c) {
                    mapping.push_back(static_cast<int>(i));
                    cls_boxes.push_back(boxes_int[i]);
                    cls_scores.push_back(scores[i]);
                }
            }
            if (cls_boxes.empty()) continue;

            std::vector<int> keep;
            cv::dnn::NMSBoxes(cls_boxes, cls_scores, CONF_THRESHOLD, IOU_THRESHOLD, keep);

            for (int ki : keep) {
                int idx = mapping[ki];
                final_dets.push_back(Detection{cls_boxes[ki], scores[idx], c});
            }
        }
        return final_dets;
    }

    // 逐帧处理
    void process_frame() {
        if (!initialized_) return;

        cv::Mat frame;
        if (!cap_.read(frame)) {
            RCLCPP_INFO(this->get_logger(), "Video processing completed. Output saved to: %s", OUTPUT_PATH.c_str());
            rclcpp::shutdown();
            return;
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // 1. 预处理
        std::vector<float> blob;
        float r;
        int pad_w, pad_h;
        preprocess(frame, blob, r, pad_w, pad_h);

        // 2. 创建输入张量
        std::array<int64_t, 4> input_shape{1, 3, INPUT_SIZE, INPUT_SIZE};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob.data(), blob.size(), input_shape.data(), input_shape.size());

        // 3. 推理
        std::vector<const char*> input_names = {input_name_.c_str()};
        std::vector<const char*> output_names = {output_name_.c_str()};
        auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

        if (output_tensors.empty() || !output_tensors.front().IsTensor()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get output tensor from ONNX session.");
            return;
        }

        // 4. 后处理
        const float* output_data = output_tensors.front().GetTensorData<float>();
        auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
        auto detections = postprocess(output_data, output_shape, r, pad_w, pad_h, frame.size());

        auto t_end = std::chrono::high_resolution_clock::now();
        float fps_now = 1000.0f / std::chrono::duration<float, std::milli>(t_end - t_start).count();

        // 5. 绘制结果
        for (const auto& det : detections) {
            cv::Scalar color = COLORS[det.class_id % COLORS.size()];
            cv::rectangle(frame, det.box, color, 2);

            std::string label = CLASS_NAMES[det.class_id] + " " + cv::format("%.2f", det.score);
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            cv::rectangle(frame, cv::Point(det.box.x, det.box.y - text_size.height - 6), cv::Point(det.box.x + text_size.width + 2, det.box.y), color, -1);
            cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 4), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
        cv::putText(frame, cv::format("FPS:%.1f", fps_now), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);

        // 6. 写入视频
        out_.write(frame);
    }

    // 成员变量
    bool initialized_;
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string output_name_;

    cv::VideoCapture cap_;
    cv::VideoWriter out_;
    rclcpp::TimerBase::SharedPtr timer_;
};

// ======================= Main 函数 =======================
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OnnxDetectNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}