//
// Created by zououming on 2022/1/21.
//

#include "ImageProcessor.h"
#include "../FeatureDetector/MySIFT.h"
using namespace cv;

ImageProcessor::ImageProcessor(const std::string& imgPath, const int& height) {
    path = imgPath;
    image = imread(imgPath);
    if(height != -1) resize(height);
    cvtColor(image, grayScale, COLOR_BGR2GRAY);
    MySIFT sift;
    sift.featurePointsCompute(image, keyPoint, descriptor);
}

void ImageProcessor::imageShow(const std::string& imgName, const int& time){
    imshow(imgName, image);
    waitKey(time);
}

Mat ImageProcessor::getImage(){
    return image.clone();
}

Mat ImageProcessor::getGrayScale(){
    return grayScale.clone();
}

void ImageProcessor::resize(int height) {
    double scale = double(height) / image.rows;
    Size size = Size(image.cols * scale, height);
    cv::resize(image, image, size);
    cvtColor(image, grayScale, COLOR_BGR2GRAY);
}

std::vector<cv::KeyPoint> ImageProcessor::getKeyPoint() {
    return keyPoint;
}

cv::Mat ImageProcessor::getDescriptor(){
    return descriptor.clone();
}

std::string ImageProcessor::getPath() {
    return path;
}
