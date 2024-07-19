#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "ball_detection.hpp"

cv::Mat segmentBallsAndField(const cv::Mat& frame, const std::vector<BoundingBox>& ball_boxes, const cv::Rect& field_rect);