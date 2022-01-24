//
// Created by zououming on 2022/1/23.
//

#ifndef IMAGESTITCHING_MATCHINGALGORITHM_H
#define IMAGESTITCHING_MATCHINGALGORITHM_H
#include "../ImageProcessor/ImageProcessor.h"
#include "opencv2/opencv.hpp"

class MatchingAlgorithm {
public:
    MatchingAlgorithm(ImageProcessor* img1, ImageProcessor* img2): imageProcessor1(img1), imageProcessor2(img2){};
    virtual void featurePointsCompute(cv::Mat& image, std::vector<cv::KeyPoint>& keyPoint, cv::Mat& descriptor) = 0;
    virtual void featurePointsMatch(const cv::Mat& descriptor1, const cv::Mat& descriptor2) = 0;
    void train(bool show = false);
    void showFeaturePoints() const;
    std::vector<cv::DMatch> getMatchPoints() const;
    std::vector<cv::KeyPoint> getKeyPoints(u_short i) const;
    ImageProcessor *imageProcessor1, *imageProcessor2;
protected:
    std::vector<cv::KeyPoint> keyPoint1, keyPoint2;
    std::vector<cv::DMatch> matchPoints; //优秀匹配点
};


#endif //IMAGESTITCHING_MATCHINGALGORITHM_H
