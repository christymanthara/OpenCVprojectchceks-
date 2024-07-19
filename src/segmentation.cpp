#include "segmentation.hpp"

cv::Mat segmentBallsAndField(const cv::Mat& frame, const std::vector<BoundingBox>& ball_boxes, const cv::Rect& field_rect) {
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);

    // Segment the playing field
    cv::rectangle(mask, field_rect, cv::Scalar(5), cv::FILLED);

    // Segment the balls
    for (const auto& box : ball_boxes) {
        cv::Rect roi(box.x, box.y, box.width, box.height);
        cv::rectangle(mask, roi, cv::Scalar(box.label), cv::FILLED);
    }

    return mask;
}