#include "evaluation.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>

double evaluateMeanAveragePrecision(const std::vector<BoundingBox>& detected_boxes, const std::string& ground_truth_path, int frame_index) {
    std::vector<BoundingBox> ground_truth_boxes;

    // Load ground truth bounding boxes from file
    std::string bbox_file = ground_truth_path + "/frame_" + std::to_string(frame_index) + "_bbox.txt";
    std::ifstream file(bbox_file);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open ground truth file: " << bbox_file << std::endl;
        return 0.0;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int x, y, width, height, label;
        if (iss >> x >> y >> width >> height >> label) {
            ground_truth_boxes.push_back({x, y, width, height, label});
        }
    }
    file.close();

    // Calculate mean Average Precision
    double mAP = calculateMeanAveragePrecision(detected_boxes, ground_truth_boxes);

    return mAP;
}

double evaluateMeanIntersectionOverUnion(const cv::Mat& segmentation_mask, const std::string& ground_truth_path, int frame_index) {
    cv::Mat ground_truth_mask;

    // Load ground truth segmentation mask from file
    std::string mask_file = ground_truth_path + "/frame_" + std::to_string(frame_index) + ".png";
    ground_truth_mask = cv::imread(mask_file, cv::IMREAD_GRAYSCALE);
    if (ground_truth_mask.empty()) {
        std::cerr << "Error: Failed to load ground truth mask: " << mask_file << std::endl;
        return 0.0;
    }

    // Calculate mean Intersection over Union
    double mIoU = calculateMeanIoU(segmentation_mask, ground_truth_mask);

    return mIoU;
}

double calculateMeanAveragePrecision(const std::vector<BoundingBox>& detected_boxes, const std::vector<BoundingBox>& ground_truth_boxes) {
    std::vector<double> precisions;
    double totalPositives = static_cast<double>(ground_truth_boxes.size());

    for (const auto& detectedBox : detected_boxes) {
        double truePositives = 0.0;
        for (const auto& groundTruthBox : ground_truth_boxes) {
            double iou = calculateIoU(detectedBox, groundTruthBox);
            if (iou >= 0.5 && detectedBox.label == groundTruthBox.label) {
                truePositives += 1.0;
            }
        }
        double precision = truePositives / (static_cast<double>(detected_boxes.size()) + std::numeric_limits<double>::epsilon());
        precisions.push_back(precision);
    }

    double mAP = 0.0;
    std::sort(precisions.begin(), precisions.end(), std::greater<double>());
    for (int i = 0; i < precisions.size(); ++i) {
        mAP += precisions[i] * ((i + 1.0) / totalPositives);
    }
    mAP /= precisions.size();

    return mAP;
}

double calculateIoU(const BoundingBox& box1, const BoundingBox& box2) {
    int xOverlap = std::max(0, std::min(box1.x + box1.width, box2.x + box2.width) - std::max(box1.x, box2.x));
    int yOverlap = std::max(0, std::min(box1.y + box1.height, box2.y + box2.height) - std::max(box1.y, box2.y));
    int intersection = xOverlap * yOverlap;
    int union_ = box1.width * box1.height + box2.width * box2.height - intersection;
    return static_cast<double>(intersection) / static_cast<double>(union_);
}

double calculateMeanIoU(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask) {
    std::vector<double> ious(6, 0.0);  // Assuming 6 classes (0-5)
    std::vector<int> total_pixels(6, 0);

    for (int i = 0; i < segmentation_mask.rows; ++i) {
        for (int j = 0; j < segmentation_mask.cols; ++j) {
            int label = static_cast<int>(segmentation_mask.at<uchar>(i, j));
            int gt_label = static_cast<int>(ground_truth_mask.at<uchar>(i, j));

            if (label == gt_label) {
                ious[label] += 1.0;
            }

            total_pixels[label] += 1;
            total_pixels[gt_label] += 1;
        }
    }

    double mIoU = 0.0;
    int num_valid_classes = 0;
    for (int i = 0; i < ious.size(); ++i) {
        if (total_pixels[i] > 0) {
            double iou = ious[i] / (total_pixels[i] + (total_pixels[i] - ious[i]));
            mIoU += iou;
            num_valid_classes += 1;
        }
    }

    if (num_valid_classes > 0) {
        mIoU /= num_valid_classes;
    }

    return mIoU;
}

std::vector<BoundingBox> loadGroundTruthBoundingBoxes(const std::string& bbox_path, int frame_number) {
    std::vector<BoundingBox> ground_truth_boxes;

    // Construct the file path for the ground truth bounding boxes
    std::string bbox_file = bbox_path + "/frame_" + std::to_string(frame_number) + "_bbox.txt";

    // Open the file
    std::ifstream file(bbox_file);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open ground truth file: " << bbox_file << std::endl;
        return ground_truth_boxes;
    }

    // Read each line from the file and parse bounding box data
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        BoundingBox bbox;
        if (iss >> bbox.x >> bbox.y >> bbox.width >> bbox.height >> bbox.label) {
            ground_truth_boxes.push_back(bbox);
        }
    }

    // Close the file
    file.close();
    return ground_truth_boxes;
}