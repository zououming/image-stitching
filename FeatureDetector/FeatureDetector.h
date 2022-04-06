//
// Created by zououming on 2022/3/3.
//

#ifndef IMAGESTITCHING_FEATUREDETECTOR_H
#define IMAGESTITCHING_FEATUREDETECTOR_H
#include <opencv2/opencv.hpp>

class FeatureDetector {
public:
    virtual void featurePointsCompute(cv::Mat image, std::vector<cv::KeyPoint>& keyPoint, cv::Mat& descriptor) = 0;
};


#endif //IMAGESTITCHING_FEATUREDETECTOR_H
