#ifndef RUNE_PACKAGE_IMAGE_PROCESSOR_HPP  
#define RUNE_PACKAGE_IMAGE_PROCESSOR_HPP
#include <opencv2/opencv.hpp>
class RuneDetector{
    public:
    enum class COLOR{BLUE,RED};
    struct Params {
        int blue_brightness_thresh = 80;
        int red_brightness_thresh = 80;
        int blue_color_thresh = 80;
        int red_color_thresh = 70;
        int morph_size = 3;
        float roi_margin_ratio = 0.1f;
        float bottom_exclude_ratio = 0.3f;
        float min_contour_area = 800.0f;
        float max_contour_area = 10000.0f;
        float min_rect_ratio = 0.6f;
        int exposure = 0;
        void updateFromTrackbars(); // 新增方法声明
    };
    COLOR enemy_color=COLOR::BLUE;
    Params param;
    cv::Mat predeal(const cv::Mat& input);
    cv::Mat findRuneArmor(const cv::Mat& input, const cv::Mat& orig);
    cv::Mat isTarget();
};
#endif