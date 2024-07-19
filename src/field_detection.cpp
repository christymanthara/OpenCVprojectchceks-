#include "field_detection.hpp"

std::vector<cv::Vec4i> detectFieldLines(const cv::Mat& frame) {
    std::vector<cv::Vec4i> lines;

    cv::Mat edges;
    cv::Canny(frame, edges, 100, 200);

    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);

    return lines;
}

cv::Rect getBoundingRect(const std::vector<cv::Vec4i>& lines) {
    std::vector<cv::Point> points;

    for (const auto& line : lines) {
        points.push_back(cv::Point(line[0], line[1]));
        points.push_back(cv::Point(line[2], line[3]));
    }

    return cv::boundingRect(points);
}