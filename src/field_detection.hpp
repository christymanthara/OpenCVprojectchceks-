#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<cv::Vec4i> detectFieldLines(const cv::Mat& frame);

cv::Rect getBoundingRect(const std::vector<cv::Vec4i>& lines);