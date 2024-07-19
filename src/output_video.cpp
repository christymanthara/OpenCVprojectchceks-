#include "output_video.hpp"

void saveOutputVideos(const std::vector<cv::Mat>& trajectories, const std::string& video_path, const std::string& output_dir) {
    cv::VideoCapture cap(video_path);
    int fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::string output_path = output_dir + "/output_video.mp4";
    cv::VideoWriter writer;
    writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height), true);

    if (!writer.isOpened()) {
        std::cerr << "Error: Failed to open output video file: " << output_path << std::endl;
        return;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        cv::Mat trajectory_overlay = frame.clone();
        for (int i = 0; i < trajectories.size(); i++) {
            cv::addWeighted(trajectory_overlay, 0.7, trajectories[i], 0.3, 0, trajectory_overlay);
            cv::rectangle(trajectory_overlay, cv::Point(10, 10 + i * 20), cv::Point(30, 30 + i * 20), cv::Scalar(255, 255, 255), -1);
        }
        writer.write(trajectory_overlay);
    }

    writer.release();
    cap.release();
}
