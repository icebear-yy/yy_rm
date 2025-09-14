#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <array>                    // 新增
#include <sensor_msgs/msg/image.hpp> // 新增
#include <cv_bridge/cv_bridge.h>     // 新增
#include <opencv2/highgui.hpp>
#include <deque> // 新增：轨迹与平滑队列
#include <opencv2/video/tracking.hpp>  // 新增：卡尔曼滤波

// ======================= 配置参数 =======================
const std::string ONNX_MODEL_PATH = "/mnt/d/rm能量机关/能量机关数据集/训练/run/exp/weights/best.onnx"; 
const std::string VIDEO_PATH = "/mnt/d/rm能量机关/能量机关视频素材（黑暗环境）/能量机关视频素材（黑暗环境）/新能量机关_正在激活.mp4";
const std::string WINDOW_NAME = "ONNX Detections"; // 新增窗口名

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
    ~OnnxDetectNode() {
        cv::destroyAllWindows();
    }

private:
    // ===== EKF（CTRV: x=[px,py,v,yaw,yaw_rate]）=====
    void ekfInit(const cv::Point2f& p, double stamp) {
        x_ = cv::Mat::zeros(5, 1, CV_32F);
        x_.at<float>(0) = p.x;  // px
        x_.at<float>(1) = p.y;  // py
        x_.at<float>(2) = 0.f;  // v
        x_.at<float>(3) = 0.f;  // yaw
        x_.at<float>(4) = 0.f;  // yaw_rate

        P_ = cv::Mat::eye(5, 5, CV_32F);
        P_.at<float>(2,2) = 10.f;
        P_.at<float>(3,3) = static_cast<float>(CV_PI*CV_PI);
        P_.at<float>(4,4) = 1.f;

        Q_ = cv::Mat::zeros(5, 5, CV_32F);
        Q_.at<float>(0,0) = 1e-2f;
        Q_.at<float>(1,1) = 1e-2f;
        Q_.at<float>(2,2) = 5e-2f;
        Q_.at<float>(3,3) = 1e-3f;
        Q_.at<float>(4,4) = 1e-3f;

        R_ = cv::Mat::eye(2, 2, CV_32F) * 4.f; // 像素量测噪声

        I5_ = cv::Mat::eye(5, 5, CV_32F);
        H_  = cv::Mat::zeros(2, 5, CV_32F);
        H_.at<float>(0,0) = 1.f; // z=[px,py]
        H_.at<float>(1,1) = 1.f;

        ekf_inited_ = true;
        last_ekf_stamp_ = stamp;
    }

    void ekfPredict(double dt) {
        dt = std::max(1e-3, dt);

        float px = x_.at<float>(0), py = x_.at<float>(1);
        float v  = x_.at<float>(2), th = x_.at<float>(3), w = x_.at<float>(4);

        // 非线性状态更新 f(x)
        const float eps = 1e-3f;
        float px_new, py_new;
        float th_new = th + w * static_cast<float>(dt);
        if (std::fabs(w) > eps) {
            float th_dt = th + w*static_cast<float>(dt);
            px_new = px + v/w * (std::sin(th_dt) - std::sin(th));
            py_new = py + v/w * (-std::cos(th_dt) + std::cos(th));
        } else {
            px_new = px + v*static_cast<float>(dt) * std::cos(th);
            py_new = py + v*static_cast<float>(dt) * std::sin(th);
        }
        x_.at<float>(0) = px_new;
        x_.at<float>(1) = py_new;
        x_.at<float>(2) = v;
        x_.at<float>(3) = th_new;
        x_.at<float>(4) = w;

        // 雅可比 F
        cv::Mat F = cv::Mat::eye(5, 5, CV_32F);
        if (std::fabs(w) > eps) {
            float th_dt = th + w*static_cast<float>(dt);
            float s_th = std::sin(th), c_th = std::cos(th);
            float s_td = std::sin(th_dt), c_td = std::cos(th_dt);
            F.at<float>(0,2) = (s_td - s_th) / w;
            F.at<float>(0,3) = v/w * (c_td - c_th);
            F.at<float>(0,4) = -v/(w*w)*(s_td - s_th) + v/w*(static_cast<float>(dt)*c_td);
            F.at<float>(1,2) = (-c_td + c_th) / w;
            F.at<float>(1,3) = v/w * (s_td - s_th);
            F.at<float>(1,4) = -v/(w*w)*(-c_td + c_th) + v/w*(static_cast<float>(dt)*s_td);
            F.at<float>(3,4) = static_cast<float>(dt);
        } else {
            F.at<float>(0,2) = static_cast<float>(dt) * std::cos(th);
            F.at<float>(0,3) = -v * static_cast<float>(dt) * std::sin(th);
            F.at<float>(1,2) = static_cast<float>(dt) * std::sin(th);
            F.at<float>(1,3) =  v * static_cast<float>(dt) * std::cos(th);
            F.at<float>(3,4) = static_cast<float>(dt);
        }

        cv::Mat Qd = Q_.clone(); Qd *= static_cast<float>(dt);
        P_ = F * P_ * F.t() + Qd;
    }

    void ekfCorrect(const cv::Point2f& meas) {
        cv::Mat z(2,1, CV_32F);
        z.at<float>(0) = meas.x; z.at<float>(1) = meas.y;

        cv::Mat hx(2,1, CV_32F);
        hx.at<float>(0) = x_.at<float>(0);
        hx.at<float>(1) = x_.at<float>(1);

        cv::Mat y = z - hx;
        cv::Mat S = H_ * P_ * H_.t() + R_;
        cv::Mat K = P_ * H_.t() * S.inv();

        x_ = x_ + K * y;
        P_ = (I5_ - K * H_) * P_;
    }

    cv::Point2f ekfExtrapolate(float dt_forward) const {
        float px = x_.at<float>(0), py = x_.at<float>(1);
        float v  = x_.at<float>(2), th = x_.at<float>(3), w = x_.at<float>(4);
        const float eps = 1e-3f;
        if (std::fabs(w) > eps) {
            float th_dt = th + w*dt_forward;
            float px2 = px + v/w * (std::sin(th_dt) - std::sin(th));
            float py2 = py + v/w * (-std::cos(th_dt) + std::cos(th));
            return {px2, py2};
        } else {
            float px2 = px + v*dt_forward*std::cos(th);
            float py2 = py + v*dt_forward*std::sin(th);
            return {px2, py2};
        }
    }

    // ====== 新增：拟合圆与预测的辅助方法 ======
    void fitCircle(const std::deque<cv::Point2f>& points, cv::Point2f& center, float& radius) {
        int n = static_cast<int>(points.size());
        if (n < 3) { center = {0,0}; radius = 0; return; }
        float sum_x=0, sum_y=0, sum_x2=0, sum_y2=0, sum_xy=0, sum_x3=0, sum_y3=0, sum_xy2=0, sum_x2y=0;
        for (const auto& p : points) {
            float x=p.x, y=p.y;
            sum_x+=x; sum_y+=y; sum_x2+=x*x; sum_y2+=y*y; sum_xy+=x*y;
            sum_x3+=x*x*x; sum_y3+=y*y*y; sum_xy2+=x*y*y; sum_x2y+=x*x*y;
        }
        float C = n * sum_x2 - sum_x * sum_x;
        float D = n * sum_xy - sum_x * sum_y;
        float E = n * sum_y2 - sum_y * sum_y;
        float G = 0.5f * (n * (sum_x3 + sum_xy2) - sum_x * (sum_x2 + sum_y2));
        float H = 0.5f * (n * (sum_y3 + sum_x2y) - sum_y * (sum_x2 + sum_y2));
        float den = C * E - D * D;
        if (std::abs(den) < 1e-6f) { center = {0,0}; radius = 0; return; }
        center.x = (E * G - D * H) / den;
        center.y = (C * H - D * G) / den;
        radius = std::sqrt((sum_x2 + sum_y2 - 2 * center.x * sum_x - 2 * center.y * sum_y + n * (center.x * center.x + center.y * center.y)) / n);
    }

    cv::Point2f predictFuturePosition(const cv::Point2f& current_position,
                                      const cv::Point2f& center,
                                      float radius,
                                      float angular_velocity,
                                      float delta_time) {
        // 使用最近两帧判断方向（沿用 image_processor 的写法）
        if (armor_center_traj_.size() < 2) return current_position;
        const auto& previous_position = armor_center_traj_[armor_center_traj_.size() - 2];

        float dx1 = current_position.x - center.x;
        float dy1 = current_position.y - center.y;
        float dx2 = previous_position.x - center.x;
        float dy2 = previous_position.y - center.y;

        float angle1 = std::atan2(dy1, dx1);
        float angle2 = std::atan2(dy2, dx2);
        float angle_diff = angle1 - angle2;
        if (angle_diff >  CV_PI) angle_diff -= 2 * CV_PI;
        if (angle_diff < -CV_PI) angle_diff += 2 * CV_PI;
        int rotation_direction = (angle_diff > 0) ? 1 : -1;

        float current_angle = angle1;
        float delta_angle = angular_velocity * delta_time * rotation_direction;
        float future_angle = current_angle + delta_angle;

        return { center.x + radius * std::cos(future_angle),
                 center.y + radius * std::sin(future_angle) };
    }

    // 初始化所有资源
    void initialize() {
        RCLCPP_INFO(this->get_logger(), "Initializing ONNX Detection Node...");

        try {
            // 1. 检查文件路径
            if (!std::filesystem::exists(ONNX_MODEL_PATH)) {
                RCLCPP_FATAL(this->get_logger(), "ONNX model file not found at: %s", ONNX_MODEL_PATH.c_str());
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

            // 3. 订阅视频话题（来自 video_publisher）
            sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                "video_frames",
                rclcpp::SensorDataQoS(),
                std::bind(&OnnxDetectNode::image_callback, this, std::placeholders::_1)
            );

            // 成功初始化后创建显示窗口
            cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

            initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Initialization successful. Waiting for frames on topic: /video_frames");

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

    // 图像回调：从话题接收图像并推理
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!initialized_) return;

        cv::Mat frame;
        try {
            frame = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed: %s", e.what());
            return;
        }
        if (frame.empty()) return;

        // 计算时间戳（优先用消息时间，否则用墙钟）
        double stamp_sec = 0.0;
        if (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) {
            stamp_sec = static_cast<double>(msg->header.stamp.sec) +
                        static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
        } else {
            static const auto t0 = std::chrono::steady_clock::now();
            stamp_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // 预处理 + 推理
        std::vector<float> blob;
        float r; int pad_w, pad_h;
        preprocess(frame, blob, r, pad_w, pad_h);
        std::array<int64_t, 4> input_shape{1, 3, INPUT_SIZE, INPUT_SIZE};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob.data(), blob.size(), input_shape.data(), input_shape.size());
        std::vector<const char*> input_names = {input_name_.c_str()};
        std::vector<const char*> output_names = {output_name_.c_str()};
        auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        if (output_tensors.empty() || !output_tensors.front().IsTensor()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get output tensor.");
            return;
        }
        const float* output_data = output_tensors.front().GetTensorData<float>();
        auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
        auto detections = postprocess(output_data, output_shape, r, pad_w, pad_h, frame.size());

        // ====== 新增：从检测结果提取 class0/1/2 的中心点 ======
        std::vector<cv::Point2f> class01_centers;
        std::vector<cv::Point2f> class2_centers;
        int best_idx_01 = -1;
        float best_score_01 = -1.f;

        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& d = detections[i];
            cv::Point2f c(d.box.x + d.box.width * 0.5f,
                          d.box.y + d.box.height* 0.5f);
            if (d.class_id == 0 || d.class_id == 1) {
                class01_centers.push_back(c);
                if (d.score > best_score_01) { best_score_01 = d.score; best_idx_01 = static_cast<int>(i); }
            } else if (d.class_id == 2) {
                class2_centers.push_back(c);
            }
        }

        // 把 class0/1 的中心加入轨迹（符页中心）
        for (const auto& c : class01_centers) {
            armor_center_traj_.push_back(c);
            if (armor_center_traj_.size() > static_cast<size_t>(100)) armor_center_traj_.pop_front();
        }

        // ====== 拟合圆心、与 class2 中心做平均 ======
        cv::Point2f final_center;
        float fitted_radius = 0.f;
        bool center_ready = false;

        if (!class01_centers.empty() && armor_center_traj_.size() >= static_cast<size_t>(kMinFitPoints_)) {
            cv::Point2f fitted_center;
            fitCircle(armor_center_traj_, fitted_center, fitted_radius);

            // 平滑圆心
            fitted_center_hist_.push_back(fitted_center);
            if (fitted_center_hist_.size() > static_cast<size_t>(kCenterSmoothLen_)) fitted_center_hist_.pop_front();
            cv::Point2f smoothed_center(0.f, 0.f);
            for (const auto& c : fitted_center_hist_) smoothed_center += c;
            smoothed_center.x /= static_cast<float>(fitted_center_hist_.size());
            smoothed_center.y /= static_cast<float>(fitted_center_hist_.size());

            // class2 平均中心
            cv::Point2f class2_avg(0.f, 0.f);
            if (!class2_centers.empty()) {
                for (const auto& c : class2_centers) class2_avg += c;
                class2_avg.x /= static_cast<float>(class2_centers.size());
                class2_avg.y /= static_cast<float>(class2_centers.size());
            }

            // 最终旋转中心
            final_center = smoothed_center;
            if (!class2_centers.empty()) {
                final_center.x = 0.5f * (smoothed_center.x + class2_avg.x);
                final_center.y = 0.5f * (smoothed_center.y + class2_avg.y);
            }
            center_ready = true;

            // 可视化
            cv::circle(frame, final_center, 8, cv::Scalar(0, 255, 255), -1);
            cv::circle(frame, final_center, static_cast<int>(fitted_radius), cv::Scalar(0, 255, 255), 2);
            cv::putText(frame, cv::format("Center:(%.0f,%.0f)", final_center.x, final_center.y),
                        final_center + cv::Point2f(12, -12), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);
        }

        // 选择当前用于预测的目标点（class0/1 里分数最高）
        cv::Point2f target_center;
        bool has_target = false;
        if (!class01_centers.empty()) {
            target_center = class01_centers.front();
            if (best_idx_01 >= 0) {
                const auto& d = detections[best_idx_01];
                target_center = { d.box.x + d.box.width * 0.5f, d.box.y + d.box.height * 0.5f };
            }
            has_target = true;
        }

        // ====== 新增：角速度稳定性判断 + EKF 分支 ======
        std::string mode_tag = "little";
        cv::Point2f future = target_center;

        if (center_ready && has_target) {
            // 更新角速度历史（用相邻两帧的夹角/时间）
            if (prev_target_valid_) {
                double dt = std::max(1e-3, stamp_sec - prev_stamp_);
                // 相对旋转中心的角度
                auto angle_of = [&](const cv::Point2f& p) {
                    return std::atan2(p.y - final_center.y, p.x - final_center.x);
                };
                float a_now = angle_of(target_center);
                float a_pre = angle_of(prev_target_);
                float da = a_now - a_pre;
                if (da >  CV_PI) da -= 2.f * CV_PI;
                if (da < -CV_PI) da += 2.f * CV_PI;
                float omega = static_cast<float>(std::fabs(da / dt));
                omega_hist_.push_back(omega);
                if (omega_hist_.size() > static_cast<size_t>(kOmegaWindow_)) omega_hist_.pop_front();
            }
            prev_target_ = target_center;
            prev_stamp_ = stamp_sec;
            prev_target_valid_ = true;

            // 统计稳定性（样本不足视为不稳定）
            bool stable = false;
            float omega_mean = 0.f, omega_std = 0.f;
            if (omega_hist_.size() >= static_cast<size_t>(kMinOmegaSamples_)) {
                for (float v : omega_hist_) omega_mean += v;
                omega_mean /= static_cast<float>(omega_hist_.size());
                float var = 0.f;
                for (float v : omega_hist_) { float d = v - omega_mean; var += d*d; }
                var /= static_cast<float>(omega_hist_.size());
                omega_std = std::sqrt(var);
                if (omega_mean > 1e-6f && (omega_std/omega_mean) <= 0.10f) stable = true;
            }

            float delta_time = 3.0f;

            if (stable) {
                float angular_velocity = (omega_mean > 0.f) ? omega_mean : (1.0f / (3.0f * CV_PI));
                future = predictFuturePosition(target_center, final_center, fitted_radius, angular_velocity, delta_time);
                mode_tag = "little";
            } else {
                // === 使用 EKF 代替原 KF ===
                if (!ekf_inited_) {
                    ekfInit(target_center, stamp_sec);
                } else {
                    double dt = std::max(1e-3, stamp_sec - last_ekf_stamp_);
                    ekfPredict(dt);
                    ekfCorrect(target_center);
                    last_ekf_stamp_ = stamp_sec;
                }
                future = ekfExtrapolate(delta_time);
                mode_tag = "big";
            }

            cv::circle(frame, future, 5, cv::Scalar(255, 0, 0), -1);
            cv::putText(frame, cv::format("Pred:(%.0f,%.0f)", future.x, future.y),
                        future + cv::Point2f(10, -10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 2);
        }

        // ====== 你原有的绘制框与FPS ======
        for (const auto& det : detections) {
            cv::Scalar color = COLORS[det.class_id % COLORS.size()];
            cv::rectangle(frame, det.box, color, 2);
            std::string label = CLASS_NAMES[det.class_id] + " " + cv::format("%.2f", det.score);
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            cv::rectangle(frame, {det.box.x, det.box.y - text_size.height - 6},
                          {det.box.x + text_size.width + 2, det.box.y}, color, -1);
            cv::putText(frame, label, {det.box.x, det.box.y - 4},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        float fps_now = 1000.0f / std::chrono::duration<float, std::milli>(t_end - t_start).count();
        cv::putText(frame, cv::format("FPS:%.1f", fps_now), {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,255,0), 2);

        // 右上角标注模式：little/big
        std::string tag_text = "mode: " + mode_tag;
        int base = 0;
        cv::Size ts = cv::getTextSize(tag_text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &base);
        cv::Point org(frame.cols - ts.width - 12, 30);
        cv::putText(frame, tag_text, org, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    (mode_tag == "big") ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0), 2);

        cv::imshow(WINDOW_NAME, frame);
        cv::waitKey(1);
    }

    // 成员变量
    bool initialized_;
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string input_name_;
    std::string output_name_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

    // ====== 新增：状态（轨迹与平滑） ======
    std::deque<cv::Point2f> armor_center_traj_;  // class0/1 的中心点轨迹
    std::deque<cv::Point2f> fitted_center_hist_; // 拟合中心的平滑历史
    const int kMinFitPoints_ = 10;
    const int kCenterSmoothLen_ = 20;

    // ====== 新增：角速度稳定性判断与卡尔曼滤波 ======
    cv::Point2f prev_target_{0.f, 0.f};
    bool prev_target_valid_ = false;
    double prev_stamp_ = 0.0;

    std::deque<float> omega_hist_;
    const int kOmegaWindow_ = 15;
    const int kMinOmegaSamples_ = 5;

    // ===== 删除旧 KF 成员，改为 EKF 成员 =====
    // cv::KalmanFilter kf_{4,2};
    // bool kf_inited_ = false;
    // double last_kf_stamp_ = 0.0;

    bool ekf_inited_ = false;
    double last_ekf_stamp_ = 0.0;
    cv::Mat x_, P_, Q_, R_, I5_, H_;
};

// ======================= Main 函数 =======================
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OnnxDetectNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}