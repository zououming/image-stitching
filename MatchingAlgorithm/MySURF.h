//
// Created by zououming on 2022/1/21.
//

#ifndef IMAGESTITCHING_MYSURF_H
#define IMAGESTITCHING_MYSURF_H
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/stitching.hpp"
#include <utility>
#include <vector>
#include "../ImageProcessor/ImageProcessor.h"
#include "MatchingAlgorithm.h"

enum precision{HIGH = 4, MID, LOW};

class MySURF: public MatchingAlgorithm{
public:
    explicit MySURF(std::vector<ImageProcessor*> images, int precision = MID):
        MatchingAlgorithm(std::move(images)), precision(precision),
        surfDetector(cv::xfeatures2d::SurfFeatureDetector::create(minHessian)) {};
    void featurePointsMatch();
    void twoImageMatch(int i, int j);
private:
    cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> surfDetector;
    int precision;
};


#endif //IMAGESTITCHING_MYSURF_H
