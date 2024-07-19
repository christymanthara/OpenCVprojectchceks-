#include "visualization.hpp"

cv::Mat visualizeTopView(const cv::Mat& frame, const cv::Mat& segmentation_mask, const cv::Rect& field_rect) {
    cv::Mat top_view = cv::Mat::zeros(field_rect.size(), CV_8UC3);

    std::vector<cv::Point2f> src_points, dst_points;
    src_points.push_back(cv::Point2f(field_rect.x, field_rect.y));
    src_points.push_back(cv::Point2f(field_rect.x + field_rect.width, field_rect.y));
    src_points.push_back(cv::Point2f(field_rect.x + field_rect.width, field_rect.y + field_rect.height));
    src_points.push_back(cv::Point2f(field_rect.x, field_rect.y + field_rect.height));

    dst_points.push_back(cv::Point2f(0, 0));
    dst_points.push_back(cv::Point2f(top_view.cols, 0));
    dst_points.push_back(cv::Point2f(top_view.cols, top_view.rows));
    dst_points.push_back(cv::Point2f(0, top_view.rows));

    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);
    cv::warpPerspective(segmentation_mask, top_view, perspective_matrix, top_view.size());

    return top_view;
}

std::vector<cv::Mat> trackBallTrajectories(const std::vector<cv::Mat>& trajectories, const cv::Mat& segmentation_mask, int frame_count) {
    std::vector<cv::Mat> updated_trajectories = trajectories;

    if (frame_count == 0) {
        updated_trajectories.clear();
        for (int i = 0; i < 6; i++) {
            updated_trajectories.push_back(cv::Mat::zeros(segmentation_mask.size(), CV_8UC1));
        }
    }

    for (int i = 0; i < segmentation_mask.rows; i++) {
        for (int j = 0; j < segmentation_mask.cols; j++) {
            int label = segmentation_mask.at<uchar>(i, j);
            if (label > 0 && label < 6) {
                updated_trajectories[label - 1].at<uchar>(i, j) = 255;
            }
        }
    }

    return updated_trajectories;
}