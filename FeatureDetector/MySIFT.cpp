//
// Created by zououming on 2022/3/3.
//

#include "MySIFT.h"
using namespace cv;

void MySIFT::featurePointsCompute(cv::Mat image, std::vector<cv::KeyPoint>& keyPoint, cv::Mat& descriptor) {
    surfDetector->detectAndCompute(image,Mat(), keyPoint, descriptor);
}
