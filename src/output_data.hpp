#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// void saveOutputData(const std::string& output_dir, 
//     const cv::Mat& first_frame, 
//     const std::vector<BoundingBox>& 
//     first_frame_ball_boxes, 
//     const cv::Mat& first_frame_segmentation_mask, 
//     const std::vector<BoundingBox>& first_frame_ground_truth_boxes, 
//     const cv::Mat& first_frame_ground_truth_mask, const cv::Mat& last_frame, 
//     const std::vector<BoundingBox>& last_frame_ball_boxes, 
//     const cv::Mat& last_frame_segmentation_mask, 
//     const std::vector<BoundingBox>& last_frame_ground_truth_boxes, 
//     const cv::Mat& last_frame_ground_truth_mask, 
//     const std::vector<cv::Mat>& trajectories, 
//     double mAP, 
//     double mIoU);

    void saveOutputData(const std::string& output_dir, const cv::Mat& first_frame, const std::vector<BoundingBox>& first_frame_ball_boxes, const cv::Mat& first_frame_segmentation_mask, const std::vector<BoundingBox>& first_frame_ground_truth_boxes, const cv::Mat& first_frame_ground_truth_mask, const cv::Mat& last_frame, const std::vector<BoundingBox>& last_frame_ball_boxes, const cv::Mat& last_frame_segmentation_mask, const std::vector<BoundingBox>& last_frame_ground_truth_boxes, const cv::Mat& last_frame_ground_truth_mask, const std::vector<cv::Mat>& trajectories, double mAP, double mIoU);
