#include "ball_detection.hpp"

BallDetector::BallDetector(const std::string& model_path) {
    net = cv::dnn::readNetFromONNX("model_path");
    class_names = {"cue_ball", "eight_ball", "solid_ball", "striped_ball"};
}

std::vector<BoundingBox> BallDetector::detect(const cv::Mat& frame) {
    std::vector<BoundingBox> boxes;

    cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    cv::Mat output = net.forward();

    float x_factor = frame.cols / 320.0;
    float y_factor = frame.rows / 320.0;

    float* data = (float*)output.data;
    for (int i = 0; i < output.rows; i++) {
        int obj_class = data[i * 6 + 1];
        float confidence = data[i * 6 + 5];

        if (confidence > 0.5) {
            int center_x = static_cast<int>(data[i * 6] * frame.cols);
            int center_y = static_cast<int>(data[i * 6 + 2] * frame.rows);
            int width = static_cast<int>(data[i * 6 + 3] * frame.cols);
            int height = static_cast<int>(data[i * 6 + 4] * frame.rows);

            int x = center_x - width / 2;
            int y = center_y - height / 2;

            BoundingBox box = {x, y, width, height, obj_class};
            boxes.push_back(box);
        }
    }

    return boxes;
}
