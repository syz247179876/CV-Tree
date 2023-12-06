#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::string projectBasePath = "/home/user/ultralytics"; // Set your ultralytics base path

    // Set this to true if you want to run inference on the GPU
    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf(projectBasePath + "/yolov8s.onnx", cv::Size(640, 480), "classes.txt", runOnGPU);

    // Pass in a vector of image paths
    std::vector<std::string> imageNames;
    imageNames.push_back(projectBasePath + "/ultralytics/assets/bus.jpg");
    imageNames.push_back(projectBasePath + "/ultralytics/assets/zidane.jpg");

    // Loop through all the images
    for (int i = 0; i < imageNames.size(); ++i)
    {
        // Read the image
        cv::Mat frame = cv::imread(imageNames[i]);

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        // Print the number of detections
        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        // Loop through all the detections
        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            // Draw the box
            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            // Detection box text
            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // This is only for preview purposes
        // Resize the image to make it easier to view
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);

        // Wait for a key press
        cv::waitKey(-1);
    }
}