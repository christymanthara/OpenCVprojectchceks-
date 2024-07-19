#include "output_data.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ball_detection.hpp"

void saveOutputData(const std::string& output_dir, const cv::Mat& first_frame, const std::vector<BoundingBox>& first_frame_ball_boxes, const cv::Mat& first_frame_segmentation_mask, const std::vector<BoundingBox>& first_frame_ground_truth_boxes, const cv::Mat& first_frame_ground_truth_mask, const cv::Mat& last_frame, const std::vector<BoundingBox>& last_frame_ball_boxes, const cv::Mat& last_frame_segmentation_mask, const std::vector<BoundingBox>& last_frame_ground_truth_boxes, const cv::Mat& last_frame_ground_truth_mask, const std::vector<cv::Mat>& trajectories, double mAP, double mIoU) {
    std::string output_video_path = output_dir + "/output_video.mp4";
    saveOutputVideos(trajectories, output_video_path);

    std::string metrics_file = output_dir + "/metrics.txt";
    std::ofstream file(metrics_file);
    if (file.is_open()) {
        file << "Mean Average Precision (mAP): " << mAP << std::endl;
        file << "Mean Intersection over Union (mIoU): " << mIoU << std::endl;
    } else {
        std::cerr << "Error: Failed to create metrics file: " << metrics_file << std::endl;
    }
    file.close();

    // Save output images (bounding boxes and segmentation masks)
    cv::imwrite(output_dir + "/first_frame_bboxes.png", drawBoundingBoxes(first_frame, first_frame_ball_boxes));
    cv::imwrite(output_dir + "/first_frame_segmentation.png", first_frame_segmentation_mask);
    cv::imwrite(output_dir + "/first_frame_ground_truth_bboxes.png", drawBoundingBoxes(first_frame, first_frame_ground_truth_boxes));
    cv::imwrite(output_dir + "/first_frame_ground_truth_segmentation.png", first_frame_ground_truth_mask);

    cv::imwrite(output_dir + "/last_frame_bboxes.png", drawBoundingBoxes(last_frame, last_frame_ball_boxes));
    cv::imwrite(output_dir + "/last_frame_segmentation.png", last_frame_segmentation_mask);
    cv::imwrite(output_dir + "/last_frame_ground_truth_bboxes.png", drawBoundingBoxes(last_frame, last_frame_ground_truth_boxes));
    cv::imwrite(output_dir + "/last_frame_ground_truth_segmentation.png", last_frame_ground_truth_mask);
}

cv::Mat drawBoundingBoxes(const cv::Mat& frame, const std::vector<BoundingBox>& bboxes) {
    cv::Mat output = frame.clone();
    for (const auto& box : bboxes) {
        cv::rectangle(output, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(0, 255, 0), 2);
    }
    return output;
}