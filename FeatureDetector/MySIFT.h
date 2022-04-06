//
// Created by zououming on 2022/3/3.
//

#ifndef IMAGESTITCHING_MYSIFT_H
#define IMAGESTITCHING_MYSIFT_H

#include "FeatureDetector.h"
#include "opencv2/xfeatures2d.hpp"

class MySIFT: public FeatureDetector{
public:
    explicit MySIFT(int minHessian = 800) {surfDetector = cv::xfeatures2d::SurfFeatureDetector::create(minHessian);};
    void featurePointsCompute(cv::Mat image, std::vector<cv::KeyPoint> &keyPoint, cv::Mat &descriptor) override;

private:
    cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> surfDetector;

};


#endif //IMAGESTITCHING_MYSIFT_H
