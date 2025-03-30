#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "rune_package/image_processor.hpp"

class RuneNode : public rclcpp::Node {
public:
    RuneNode() : Node("rune_detector") {
        // 创建参数调节窗口
        cv::namedWindow("Params", cv::WINDOW_NORMAL);
        cv::resizeWindow("Params", 500, 300);
        
        // 初始化滑块
        createTrackbars();
        
        // 创建图像订阅
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "input_image", 10, std::bind(&RuneNode::imageCallback, this, std::placeholders::_1));
    }

    ~RuneNode() {
        cv::destroyAllWindows();
    }

private:
    void createTrackbars() {
        // 初始化滑块（参数顺序：名称，窗口名，值指针，最大值）
        cv::createTrackbar("Blue_Bright", "Params", &detector_.param.blue_brightness_thresh, 255);
        cv::createTrackbar("Blue_Color", "Params", &detector_.param.blue_color_thresh, 255);
        cv::createTrackbar("Red_Bright", "Params", &detector_.param.red_brightness_thresh, 255);
        cv::createTrackbar("Red_Color", "Params", &detector_.param.red_color_thresh, 255);
        cv::createTrackbar("Morph_Size", "Params", &detector_.param.morph_size, 20);
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // 转换图像
            cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
            
            // 处理图像（自动使用滑块的最新参数值）
            detector_.findRuneArmor(frame, frame);
            
            // 显示处理结果
            cv::imshow("Processed", frame);
            cv::waitKey(1);  // 必须调用才能更新滑块值
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "图像转换错误: %s", e.what());
        }
    }

    RuneDetector detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RuneNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}