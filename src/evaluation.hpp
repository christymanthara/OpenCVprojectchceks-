#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include "ball_detection.hpp"


// struct BoundingB {
//     int x, y, width, height;
//     int label;
// };

std::vector<BoundingBox> loadGroundTruthBoundingBoxes(const std::string& bbox_path, int frame_number);

double calculateMeanAveragePrecision(
    const std::vector<BoundingBox>& predicted_boxes,
    const std::vector<BoundingBox>& ground_truth_boxes);

double evaluateMeanAveragePrecision(
    const std::vector<BoundingBox>& predicted_boxes,
    const std::vector<BoundingBox>& ground_truth_boxes,
    int frame_number);

double evaluateMeanIntersectionOverUnion(
    const cv::Mat& segmentation_mask,
    const cv::Mat& ground_truth_mask,
    int frame_number);

double calculateMeanIoU(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask);
double calculateIoU(const BoundingBox& boxA, const BoundingBox& boxB);

std::vector<BoundingBox> loadGroundTruthBoundingBoxes(const std::string& bbox_path, int frame_number);
