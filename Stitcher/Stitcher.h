//
// Created by zououming on 2022/1/23.
//

#ifndef IMAGESTITCHING_STITCHER_H
#define IMAGESTITCHING_STITCHER_H
#include "../MatchingAlgorithm/MatchingAlgorithm.h"
#include "../ImageProcessor/ImageProcessor.h"
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d.hpp"

class Stitcher {
    int midIndex, optMethod;
    cv::Mat stitchedImage;
    std::vector<cv::Mat> homographyMats, transformImages;
    std::vector<std::vector<cv::Point>> corners;
    cv::Size resultSize;
    inline double easeInOutOpt(double);
public:
    explicit Stitcher(MatchingAlgorithm& algorithm, int optMethod = easeInOut): algorithm(&algorithm), optMethod(optMethod){};
    void transform();
    int8_t homographyCompute();
    std::vector<cv::Point> cornersCompute(const int& index);
    void imageStitch();
    void resultSizeCompute(int midHeight);
    void optimizeSeam(int, int);
    void train(const bool& show);
    cv::Mat getResult();
    MatchingAlgorithm* algorithm;
    enum {leftTop, leftBottom, rightTop, rightBottom};
    enum {liner, easeInOut};
};


#endif //IMAGESTITCHING_STITCHER_H
