#include<rclcpp/rclcpp.hpp>
#include<sensor_msgs/msg/image.hpp>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/opencv.hpp>
#include"rune_package/image_processor.hpp"

class VideoSubscriber : public rclcpp::Node {
public:
    VideoSubscriber() : Node("video_subscriber") {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "video_frames", 10,
            std::bind(&VideoSubscriber::imageCallback, this, std::placeholders::_1));
    }
private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv::Mat frame=cv_bridge::toCvCopy(msg, "bgr8")->image;
            cv::Mat processed=detector_.predeal(frame);
            cv::Mat final=detector_.findRuneArmor(processed,frame);
            cv::imshow("Received Video",frame);
            cv::imshow("Processed Result",processed);
            cv::imshow("final Result",final);
            cv::waitKey(1);
        }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge 异常: %s", e.what());
        }
    }
    RuneDetector detector_; 
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoSubscriber>());
    rclcpp::shutdown();
    return 0;
}