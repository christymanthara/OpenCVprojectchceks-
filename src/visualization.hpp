#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat visualizeTopView(const cv::Mat& frame, const cv::Mat& segmentation_mask, const cv::Rect& field_rect);
std::vector<cv::Mat> trackBallTrajectories(const std::vector<cv::Mat>& trajectories, const cv::Mat& segmentation_mask, int frame_count);