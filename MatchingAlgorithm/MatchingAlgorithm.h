//
// Created by zououming on 2022/1/23.
//

#ifndef IMAGESTITCHING_MATCHINGALGORITHM_H
#define IMAGESTITCHING_MATCHINGALGORITHM_H
#include <utility>

#include "../ImageProcessor/ImageProcessor.h"
#include "opencv2/opencv.hpp"

class MatchingAlgorithm {
    cv::FlannBasedMatcher matcher;
    void findMaxIJ(const std::vector<std::vector<int>>& matrix, int& resI, int& resJ);
    std::vector<std::vector<bool>> kruskal(std::vector<std::vector<int>>);
    int findCenter(const std::vector<std::vector<int>> &);
    int midIndex;
public:
    explicit MatchingAlgorithm(std::vector<ImageProcessor*> images, int8_t precision = 5): images(std::move(images)), precision(precision){};
    void sortImage(std::vector<ImageProcessor*>&); //去除多余图像
    int sortImageByPosition(std::vector<ImageProcessor*>&);
    void featurePointsMatch();
    void train(bool show = false);
    void showFeaturePoints(const std::vector<ImageProcessor*>&, const std::vector<std::vector<cv::DMatch>>&);
    std::vector<cv::DMatch> twoImageMatch(int i, int j);
    std::vector<cv::DMatch> getMatchPoints(int i) const;
    std::vector<ImageProcessor*> images;
    enum precision{HIGH = 4, MID, LOW};
protected:
    int8_t precision;
    std::vector<std::vector<cv::DMatch>> matchPointsList; //优秀匹配点
};


#endif //IMAGESTITCHING_MATCHINGALGORITHM_H