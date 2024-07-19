#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

struct BoundingBox {
    int x, y, width, height;
    int label;
};

class BallDetector {
public:
    BallDetector(const std::string& model_path);
    std::vector<BoundingBox> detect(const cv::Mat& frame);

private:
    cv::dnn::Net net;
    std::vector<std::string> class_names;
};