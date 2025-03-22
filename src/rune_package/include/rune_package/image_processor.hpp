#ifndef RUNE_PACKAGE_IMAGE_PROCESSOR_HPP  
#define RUNE_PACKAGE_IMAGE_PROCESSOR_HPP
#include <opencv2/opencv.hpp>
class RuneDetector{
    public:
    enum class COLOR{BLUE,RED};
    struct Params{
        int blue_brightness_thresh=70;
        int blue_color_thresh=60;
        int red_brightness_thresh=80;
        int red_color_thresh=70;
        int morph_size=3;
    };
    COLOR enemy_color=COLOR::RED;
    Params param;
    cv::Mat predeal(const cv::Mat& input);
    cv::Mat findRuneArmor(const cv::Mat& input, const cv::Mat& orig);
};
#endif