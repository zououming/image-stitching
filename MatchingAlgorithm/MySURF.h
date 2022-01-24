//
// Created by zououming on 2022/1/21.
//

#ifndef IMAGESTITCHING_MYSURF_H
#define IMAGESTITCHING_MYSURF_H
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <vector>
#include "../ImageProcessor/ImageProcessor.h"
#include "MatchingAlgorithm.h"

enum precision{HIGH = 4, MID, LOW};

class MySURF: public MatchingAlgorithm{
public:
    MySURF(ImageProcessor* img1, ImageProcessor* img2, int minHessian = 800, int precision = MID):
        MatchingAlgorithm(img1, img2), precision(precision),
        surfDetector(cv::xfeatures2d::SurfFeatureDetector::create(minHessian)) {};
    void featurePointsCompute(cv::Mat& image, std::vector<cv::KeyPoint>& keyPoint, cv::Mat& descriptor) override;
    void featurePointsMatch(const cv::Mat& descriptor1, const cv::Mat& descriptor2) override;
private:
    cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> surfDetector;
    int precision;
};


#endif //IMAGESTITCHING_MYSURF_H
