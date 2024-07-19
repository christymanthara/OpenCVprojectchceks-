#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

void saveOutputVideos(const std::vector<cv::Mat>& trajectories, const std::string& video_path, const std::string& output_dir);
