#include<rclcpp/rclcpp.hpp>
#include<sensor_msgs/msg/image.hpp>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<vector>
#include"rune_package/image_processor.hpp"
using namespace std;
using namespace cv;



void RuneDetector::Params::updateFromTrackbars() {
    static bool trackbars_created = false; 

    if (trackbars_created) {
        return; 
    }

    cv::namedWindow("Parameter Adjustment", cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("Blue Brightness", "Parameter Adjustment", &blue_brightness_thresh, 255);
    cv::createTrackbar("Red Brightness", "Parameter Adjustment", &red_brightness_thresh, 255);
    cv::createTrackbar("Blue Color", "Parameter Adjustment", &blue_color_thresh, 255);
    cv::createTrackbar("Red Color", "Parameter Adjustment", &red_color_thresh, 255);
    cv::createTrackbar("Morph Size", "Parameter Adjustment", &morph_size, 20);
    cv::createTrackbar("ROI Margin (%)", "Parameter Adjustment", nullptr, 100,
        [](int value, void* userdata) {
            auto* params = static_cast<RuneDetector::Params*>(userdata);
            if (params) {
                params->roi_margin_ratio = value / 100.0f;
            }
        }, this);
    cv::createTrackbar("Bottom Exclude (%)", "Parameter Adjustment", nullptr, 100,
        [](int value, void* userdata) {
            auto* params = static_cast<RuneDetector::Params*>(userdata);
            if (params) {
                params->bottom_exclude_ratio = value / 100.0f;
            }
        }, this);
    cv::createTrackbar("Exposure", "Parameter Adjustment", &exposure, 100);

    trackbars_created = true; 
}


Mat RuneDetector::predeal(const Mat& input) {
    Mat color_diff, gray, binary_result;
    vector<Mat> bgr_channels;
    Mat adjusted_input;
    
    double alpha = 1;
    int beta = param.exposure;
    convertScaleAbs(input, adjusted_input, alpha, beta);
    cvtColor(adjusted_input, gray, COLOR_BGR2GRAY);
    split(adjusted_input, bgr_channels);

    if (enemy_color == COLOR::BLUE) {
        subtract(bgr_channels[0], bgr_channels[2], color_diff);
        threshold(gray, gray, param.blue_brightness_thresh, 255, THRESH_BINARY);
        threshold(color_diff, color_diff, param.blue_color_thresh, 255, THRESH_BINARY);
    } else {
        subtract(bgr_channels[2], bgr_channels[0], color_diff);
        threshold(gray, gray, param.red_brightness_thresh, 255, THRESH_BINARY);
        threshold(color_diff, color_diff, param.red_color_thresh, 255, THRESH_BINARY);
    }
    
    bitwise_and(gray, color_diff, binary_result);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(param.morph_size, param.morph_size));
    morphologyEx(binary_result, binary_result, MORPH_OPEN, kernel);
    return binary_result;
}

Mat RuneDetector::findRuneArmor(const Mat& input, const Mat& orig) {
    Mat result = orig.clone();
    int roi_margin_x = static_cast<int>(input.cols * param.roi_margin_ratio);
    int roi_margin_y = static_cast<int>(input.rows * param.roi_margin_ratio);
    Rect roi_rect(
        roi_margin_x, roi_margin_y,
        input.cols - 2 * roi_margin_x, input.rows - 2 * roi_margin_y);
    int bottom_exclude = static_cast<int>(input.rows * param.bottom_exclude_ratio);
    vector<vector<Point>> contours;
    findContours(input, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Point2f> armor_centers;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < param.min_contour_area || area > param.max_contour_area) continue;
        RotatedRect rect = minAreaRect(contour);
        float rect_ratio = area / rect.size.area();
        if (rect_ratio < param.min_rect_ratio) continue;
        if (!roi_rect.contains(rect.center)) continue;
        if (rect.center.y > (input.rows - bottom_exclude)) continue;
        armor_centers.push_back(rect.center);
    }
    vector<Point2f> all_points;
    for (int y = roi_rect.y; y < roi_rect.y + roi_rect.height; ++y) {
        for (int x = roi_rect.x; x < roi_rect.x + roi_rect.width; ++x) {
            if (input.at<uchar>(y, x) == 255) {
                all_points.emplace_back(x, y);
            }
        }
    }
    if (!all_points.empty()) {
        Point2f rotation_center;
        float radius;
        minEnclosingCircle(all_points, rotation_center, radius);
        circle(result, rotation_center, 8, Scalar(255, 0, 0), -1);
        putText(result, format("Center: (%.0f,%.0f)", rotation_center.x, rotation_center.y),
                rotation_center + Point2f(15, -15), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        
        for (const auto& center : armor_centers) {
            circle(result, center, 5, Scalar(0, 255, 0), -1);
            line(result, center, rotation_center, Scalar(0, 200, 200), 1);
            putText(result, format("(%.0f,%.0f)", center.x, center.y),
                    center + Point2f(10, 5), FONT_HERSHEY_PLAIN, 1.4, Scalar(0, 200, 0), 2);
        }
    }
    return result;
}





