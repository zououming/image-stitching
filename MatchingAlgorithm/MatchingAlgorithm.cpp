//
// Created by zououming on 2022/1/23.
//

#include "MatchingAlgorithm.h"

void MatchingAlgorithm::showFeaturePoints() const {
    cv::Mat imgMatches;
    drawMatches(*imageProcessor2->getImage(), keyPoint2,
                *imageProcessor1->getImage(), keyPoint1, matchPoints, imgMatches);//进行绘制
    imshow("匹配图", imgMatches);
    cv::waitKey(0);
}

std::vector<cv::DMatch> MatchingAlgorithm::getMatchPoints() const {
    return matchPoints;
}

void MatchingAlgorithm::train(bool show) {
    cv::Mat descriptor1, descriptor2;
    featurePointsCompute(*imageProcessor1->getGrayScale(), keyPoint1, descriptor1);
    featurePointsCompute(*imageProcessor2->getGrayScale(), keyPoint2, descriptor2);
    featurePointsMatch(descriptor1, descriptor2);
    if(show)
        showFeaturePoints();
}

std::vector<cv::KeyPoint> MatchingAlgorithm::getKeyPoints(u_short i) const {
    if(i == 1)
        return keyPoint1;
    if(i == 2)
        return keyPoint2;
}
