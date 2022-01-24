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
public:
    Stitcher(MatchingAlgorithm* algorithm): matchingAlgorithm(algorithm),
        imageProcessor1{algorithm->imageProcessor1}, imageProcessor2{algorithm->imageProcessor2}{};
    void transform();
    MatchingAlgorithm* matchingAlgorithm;
    ImageProcessor *imageProcessor1, *imageProcessor2;
private:
    cv::Mat transformMat1, transformMat2;
};


#endif //IMAGESTITCHING_STITCHER_H
