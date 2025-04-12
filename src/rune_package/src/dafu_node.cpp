#include<rclcpp/rclcpp.hpp>
#include<sensor_msgs/msg/image.hpp>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/opencv.hpp>
#include <thread> 
#include <atomic>
#include <mutex> // 添加互斥锁支持
#include"rune_package/image_processor.hpp"


class VideoSubscriber : public rclcpp::Node {
public:
    VideoSubscriber() : Node("video_subscriber"), stop_thread_(false) {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "video_frames", 10,
            std::bind(&VideoSubscriber::imageCallback, this, std::placeholders::_1));

        slider_thread_ = std::thread(&VideoSubscriber::runSliderWindow, this);
    }

    ~VideoSubscriber() {
        stop_thread_ = true;
        if (slider_thread_.joinable()) {
            slider_thread_.join();
        }
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            std::lock_guard<std::mutex> lock(param_mutex_); // 加锁
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            cv::Mat processed = detector_.predeal(frame);
            cv::Mat final = detector_.findRuneArmor(processed, frame);

            cv::imshow("Processed Result", processed);
            cv::imshow("Final Result", final);
            cv::waitKey(1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge 异常: %s", e.what());
        }
    }

    void runSliderWindow() {
        detector_.param.updateFromTrackbars(); // 确保滑块只创建一次

        while (!stop_thread_) {
            {
                std::lock_guard<std::mutex> lock(param_mutex_); // 加锁
                // 参数值会通过滑块实时更新，无需重复调用
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    RuneDetector detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    std::thread slider_thread_;
    std::atomic<bool> stop_thread_;
    std::mutex param_mutex_; // 互斥锁
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoSubscriber>());
    rclcpp::shutdown();
    return 0;
}