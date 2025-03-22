#include<rclcpp/rclcpp.hpp>
#include<sensor_msgs/msg/image.hpp>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/opencv.hpp>
#include<vector>
#include"rune_package/image_processor.hpp"
using namespace std;
using namespace cv;
Mat RuneDetector::predeal(const Mat& input){
    Mat color_diff,gray,binary_result;
    vector<Mat>bgr_channels;
    Mat adjusted_input;
    double alpha=1;
    convertScaleAbs(input,adjusted_input,alpha,0);
    cvtColor(adjusted_input,gray,COLOR_BGR2GRAY);
    split(adjusted_input, bgr_channels);
    if (enemy_color==COLOR::BLUE) {
        subtract(bgr_channels[0],bgr_channels[2],color_diff);
        threshold(gray,gray,param.blue_brightness_thresh,255,THRESH_BINARY);
        threshold(color_diff,color_diff,param.blue_color_thresh,255,THRESH_BINARY);
    } else {
        subtract(bgr_channels[2],bgr_channels[0],color_diff);
        threshold(gray,gray,param.red_brightness_thresh,255,THRESH_BINARY);
        threshold(color_diff,color_diff,param.red_color_thresh,255,THRESH_BINARY);
    }
    bitwise_and(gray,color_diff,binary_result);
    Mat kernel=getStructuringElement(MORPH_RECT,Size(3,3));
    morphologyEx(binary_result,binary_result,MORPH_OPEN,kernel);
    return binary_result;
};
Mat RuneDetector::findRuneArmor(const Mat&input,const Mat&orig) {
    Mat result=orig.clone(); 
    const float ROI_MARGIN_RATIO=0.1f;
    int roi_margin_x=static_cast<int>(input.cols*ROI_MARGIN_RATIO);
    int roi_margin_y=static_cast<int>(input.rows*ROI_MARGIN_RATIO);
    Rect roi_rect(
        roi_margin_x, roi_margin_y,
        input.cols-2*roi_margin_x,input.rows-2*roi_margin_y);
    const float BOTTOM_EXCLUDE_RATIO=0.30f; 
    int bottom_exclude = static_cast<int>(input.rows * BOTTOM_EXCLUDE_RATIO);
    vector<vector<Point>> contours;
    findContours(input,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    vector<Point2f>armor_centers;
    const float MIN_AREA=800.0f;
    const float MAX_AREA=10000.0f;
    const float MIN_RECT_RATIO=0.3f; 

    for(const auto& contour : contours){
        double area=contourArea(contour);
        if(area<MIN_AREA||area>MAX_AREA)continue;
        RotatedRect rect=minAreaRect(contour);
        float rect_ratio=area/rect.size.area();
        if(rect_ratio<MIN_RECT_RATIO)continue;
        if(!roi_rect.contains(rect.center))continue;
        if(rect.center.y>(input.rows-bottom_exclude))continue;
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
        rotation_center.y -= 45;
        circle(result, rotation_center, 8, Scalar(255, 0, 0), -1); 
        putText(result, format("Center: (%.0f,%.0f)", rotation_center.x, rotation_center.y),
                rotation_center + Point2f(15, -15), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
        for (const auto& center : armor_centers) {
            circle(result,center,5,Scalar(0,255,0),-1);
            line(result,center,rotation_center,Scalar(0,200,200),1);
            putText(result, format("(%.0f,%.0f)",center.x,center.y),
                      center+Point2f(10,5),FONT_HERSHEY_PLAIN,1.4,Scalar(0,200,0),2);
        }
    }
    return result; 
}
