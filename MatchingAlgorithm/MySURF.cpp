//f
// Created by zououming on 2022/1/21.
//

#include "MySURF.h"
using namespace std;
using namespace cv;
using namespace xfeatures2d;

void MySURF::featurePointsMatch() {
    twoImageMatch(0, 1);
}

void MySURF::twoImageMatch(int i, int j) {
    FlannBasedMatcher matcher;
    std::vector<std::vector<DMatch>> matchRes;

    std::vector<Mat> train_desc(1, images[i]->getDescriptor());
    matcher.add(train_desc);
    matcher.train();
    matcher.knnMatch(images[j]->getDescriptor(), matchRes, 2);

    double threshold = double(precision) / 10;
    for(auto& match : matchRes)
        if(match[0].distance < match[1].distance * threshold)
            matchPoints.push_back(match[0]);
}
