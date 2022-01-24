//
// Created by zououming on 2022/1/21.
//

#include "MySURF.h"
using namespace std;
using namespace cv;
using namespace xfeatures2d;

void MySURF::featurePointsCompute(cv::Mat& image, std::vector<cv::KeyPoint>& keyPoint, cv::Mat& descriptor) {
    surfDetector->detectAndCompute(image,Mat(), keyPoint, descriptor);
}

void MySURF::featurePointsMatch(const cv::Mat& descriptor1,const cv::Mat& descriptor2) {
    FlannBasedMatcher matcher;
    std::vector<std::vector<DMatch>> matchRes;

    std::vector<Mat> train_desc(1, descriptor1);
    matcher.add(train_desc);
    matcher.train();
    matcher.knnMatch(descriptor2, matchRes, 2);

    double threshold = double(precision) / 10;
    for(auto& match : matchRes)
        if(match[0].distance < match[1].distance * threshold)
            matchPoints.push_back(match[0]);
}