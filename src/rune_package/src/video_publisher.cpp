#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class VideoPublisher : public rclcpp::Node {
public:
    VideoPublisher() : Node("video_publisher") {
        if (!cap_.open("/mnt/d/rm能量机关/能量机关视频素材（黑暗环境）/能量机关视频素材（黑暗环境）/新能量机关_正在激活.mp4")) {
            RCLCPP_ERROR(this->get_logger(), "无法打开视频文件！");
            rclcpp::shutdown();
            return;
        }
        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("video_frames", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), 
            std::bind(&VideoPublisher::publishFrame, this)
        );
    }

private:
    void publishFrame() {
        cv::Mat frame;
        cap_ >> frame; 

        if (frame.empty()) {
            RCLCPP_INFO(this->get_logger(), "视频处理完成");
            rclcpp::shutdown();
            return;
        }
        auto msg = cv_bridge::CvImage(
            std_msgs::msg::Header(), "bgr8", frame
        ).toImageMsg();
        publisher_->publish(*msg);
        // cv::imshow("Video Frame", frame);
        cv::waitKey(1);
    }

    cv::VideoCapture cap_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VideoPublisher>();
    rclcpp::spin(node);
    return 0;
}



