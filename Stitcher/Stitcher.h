//
// Created by zououming on 2022/1/23.
//

#ifndef IMAGESTITCHING_STITCHER_H
#define IMAGESTITCHING_STITCHER_H
#include "../MatchingAlgorithm/MatchingAlgorithm.h"
#include "../ImageProcessor/ImageProcessor.h"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

class Stitcher {
    int midIndex, optMethod;
    cv::Mat stitchedImage;
    std::vector<cv::Mat> homographyMats, transformImages;
    cv::Size resultSize;
    inline double easeInOutOpt(double);
    std::vector<cv::Point> overlappingAreaCompute(std::vector<cv::Point>, std::vector<cv::Point>);
    std::vector<cv::Point> boundingRectangleCompute(const std::vector<cv::Point>& corner);
    std::vector<double> fitLine(const std::vector<uchar>&, const std::vector<uchar>&);
public:
    explicit Stitcher(MatchingAlgorithm& algorithm, int optMethod = easeInOut): algorithm(&algorithm), optMethod(optMethod){};
    void transform(const int&, const cv::Size&);
    int8_t homographyCompute(const bool&);
    std::vector<cv::Point> cornersCompute(const int& index);
    void imageStitch(int, cv::Mat&);
    cv::Size resultSizeCompute();
    void ExposureAdjustment();
    void imageFuse(const int&, const int&, cv::Point, cv::Point, const int&);
    void optimizeSeam(const cv::Point&, const cv::Point&, cv::Mat&);
    void train(const bool& show);
    cv::Mat getResult();
    MatchingAlgorithm* algorithm;
    enum {leftTop, leftBottom, rightTop, rightBottom};
    enum {liner, easeInOut};
};


#endif //IMAGESTITCHING_STITCHER_H
