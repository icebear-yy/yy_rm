#include<rclcpp/rclcpp.hpp>
#include<sensor_msgs/msg/image.hpp>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<vector>
#include<filesystem>
#include<iomanip>
#include <chrono>
#include<sstream>
#include"rune_package/image_processor.hpp"
using namespace std;
using namespace cv;


//接收图像
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

//预处理
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


//寻找符页
Mat RuneDetector::findRuneArmor(const Mat& input, const Mat& orig) {
    Mat result = orig.clone();
    Point2f target_center(-1, -1);  

    int roi_margin_x = static_cast<int>(orig.cols * param.roi_margin_ratio);
    int roi_margin_y = static_cast<int>(orig.rows * param.roi_margin_ratio);
    Rect roi_rect(
        roi_margin_x, roi_margin_y,
        orig.cols - 2 * roi_margin_x, orig.rows - 2 * roi_margin_y);
    int bottom_exclude = static_cast<int>(orig.rows * param.bottom_exclude_ratio);

    vector<vector<Point>> contours;
    findContours(input, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Point2f> armor_centers;
    vector<pair<double, Point2f>> area_center_pairs; 

    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area < param.min_contour_area || area > param.max_contour_area) continue;
        RotatedRect rect = minAreaRect(contour);
        float rect_ratio = area / rect.size.area();
        if (rect_ratio < param.min_rect_ratio) continue;
        if (!roi_rect.contains(rect.center)) continue;
        if (rect.center.y > (orig.rows - bottom_exclude)) continue;

        Point2f vertices[4];
        rect.points(vertices);
        for (int i = 0; i < 4; ++i) {
            circle(result, vertices[i], 3, Scalar(0, 0, 255), -1);
            putText(result, 
                format("(%.0f,%.0f)", vertices[i].x, vertices[i].y),
                vertices[i] + Point2f(5, 5),
                FONT_HERSHEY_PLAIN, 
                0.8, 
                Scalar(0, 0, 255),
                1);
        }
        armor_centers.push_back(rect.center);
        area_center_pairs.emplace_back(area, rect.center);
    }

    for (const auto& center : armor_centers) {
        armor_center_trajectory.push_back(center);
        if (armor_center_trajectory.size() > 100) {
            armor_center_trajectory.pop_front();
        }
    }

    //判断是否为目标中心，使用是否有子轮廓
    vector<vector<Point>> children;
    vector<Vec4i> hierarchy;
    findContours(input, children, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < children.size(); ++i) {
        if (hierarchy[i][2] != -1) { 
            RotatedRect rect = minAreaRect(children[i]);
            target_center = rect.center;
            break;
        }
    }

//拟合旋转圆心
    if (armor_center_trajectory.size() >= 10 && target_center.x >= 0) {
        Point2f fitted_center;
        float fitted_radius;
        fitCircle(armor_center_trajectory, fitted_center, fitted_radius);
        fitted_center_history.push_back(fitted_center);
        if (fitted_center_history.size() > 20) {
            fitted_center_history.pop_front();
        }

        Point2f smoothed_center(0, 0);
        for (const auto& center : fitted_center_history) {
            smoothed_center += center;
        }
        smoothed_center.x /= fitted_center_history.size();
        smoothed_center.y /= fitted_center_history.size();

        circle(result, smoothed_center, 8, Scalar(0, 255, 255), -1);
        circle(result, smoothed_center, static_cast<int>(fitted_radius), Scalar(0, 255, 255), 2);
        putText(result, format("Smoothed Center: (%.0f,%.0f)", smoothed_center.x, smoothed_center.y),
                smoothed_center + Point2f(15, -15), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);

        Point2f future_position = predictFuturePosition(
            target_center, smoothed_center, fitted_radius, 1.0f / (3 * CV_PI), 3.0f);

        circle(result, future_position, 5, Scalar(255, 0, 0), -1);
        putText(result, format("Predicted: (%.0f,%.0f)", future_position.x, future_position.y),
                future_position + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
    }

    //绘制装甲板中心
    for (const auto& center : armor_centers) {
        bool is_target = norm(center - target_center) < 1e-3;
        Scalar point_color = is_target ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
        Scalar text_color  = is_target ? Scalar(0, 0, 255) : Scalar(0, 200, 0);
        circle(result, center, 5, point_color, -1);
        putText(result, format("(%.0f,%.0f)", center.x, center.y),
                center + Point2f(10, 5), FONT_HERSHEY_PLAIN, 1.4, text_color, 2);
            }

    return result;
}

//拟合圆心方法
void fitCircle(const std::deque<cv::Point2f>& points, cv::Point2f& center, float& radius) {
    int n = points.size();
    if (n < 3) {
        center = cv::Point2f(0, 0);
        radius = 0;
        return;
    }
    float sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0, sum_xy = 0, sum_x3 = 0, sum_y3 = 0, sum_xy2 = 0, sum_x2y = 0;
    for (const auto& p : points) {
        float x = p.x, y = p.y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        sum_xy += x * y;
        sum_x3 += x * x * x;
        sum_y3 += y * y * y;
        sum_xy2 += x * y * y;
        sum_x2y += x * x * y;
    }
    float C = n * sum_x2 - sum_x * sum_x;
    float D = n * sum_xy - sum_x * sum_y;
    float E = n * sum_y2 - sum_y * sum_y;
    float G = 0.5 * (n * (sum_x3 + sum_xy2) - sum_x * (sum_x2 + sum_y2));
    float H = 0.5 * (n * (sum_y3 + sum_x2y) - sum_y * (sum_x2 + sum_y2));
    float denominator = C * E - D * D;
    if (std::abs(denominator) < 1e-6) {
        center = cv::Point2f(0, 0);
        radius = 0;
        return;
    }
    center.x = (E * G - D * H) / denominator;
    center.y = (C * H - D * G) / denominator;
    radius = std::sqrt((sum_x2 + sum_y2 - 2 * center.x * sum_x - 2 * center.y * sum_y + n * (center.x * center.x + center.y * center.y)) / n);
}


//预测未来位置
cv::Point2f RuneDetector::predictFuturePosition(const cv::Point2f& current_position, const cv::Point2f& center, float radius, float angular_velocity, float delta_time) {
    // 判断旋转方向
    if (armor_center_trajectory.size() < 2) {
        // 如果轨迹点不足，无法判断方向，直接返回当前点
        return current_position;
    }
    // 获取当前点和前一个点
    const cv::Point2f& previous_position = armor_center_trajectory[armor_center_trajectory.size() - 2];
    float dx1 = current_position.x - center.x;
    float dy1 = current_position.y - center.y;
    float dx2 = previous_position.x - center.x;
    float dy2 = previous_position.y - center.y;
    // 计算当前点和前一个点的角度
    float angle1 = atan2(dy1, dx1); // 当前点相对于圆心的角度
    float angle2 = atan2(dy2, dx2); // 前一个点相对于圆心的角度
    // 计算角度差
    float angle_diff = angle1 - angle2;
    // 规范化角度差到 [-π, π]
    if (angle_diff > CV_PI) {
        angle_diff -= 2 * CV_PI;
    } else if (angle_diff < -CV_PI) {
        angle_diff += 2 * CV_PI;
    }
    // 判断旋转方向
    int rotation_direction = (angle_diff > 0) ? 1 : -1; // 逆时针为 1，顺时针为 -1
    // 计算当前角度
    float current_angle = atan2(dy1, dx1); // 当前角度（弧度）
    // 计算未来角度
    float delta_angle = angular_velocity * delta_time * rotation_direction; // 根据方向调整角度变化
    float future_angle = current_angle + delta_angle;
    // 计算未来位置
    float future_x = center.x + radius * cos(future_angle);
    float future_y = center.y + radius * sin(future_angle);
    return cv::Point2f(future_x, future_y);
}