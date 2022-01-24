//
// Created by zououming on 2022/1/23.
//

#include "Stitcher.h"
using namespace std;

void Stitcher::transform() {
    vector<cv::Point2f> imagePoints1, imagePoints2;
    vector<cv::KeyPoint> keyPoint1, keyPoint2;
    keyPoint1 = matchingAlgorithm->getKeyPoints(1);
    keyPoint2 = matchingAlgorithm->getKeyPoints(2);
    for(auto& match : matchingAlgorithm->getMatchPoints()){
        imagePoints1.push_back(keyPoint2[match.trainIdx].pt);
        imagePoints2.push_back(keyPoint1[match.queryIdx].pt);
    }
    cv::Mat homo = cv::findHomography(imagePoints1, imagePoints2);
//    cv::warpPerspective(*imageProcessor1->getImage(), transformMat1, homo, cv::Size(max()))
}
