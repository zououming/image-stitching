//
// Created by zououming on 2022/1/21.
//

#include "ImageProcessor.h"
using namespace cv;

ImageProcessor::ImageProcessor(const std::string& imgPath) {
    image = imread(imgPath);
    cvtColor(image, grayscale, COLOR_BGR2GRAY);
}

void ImageProcessor::imageShow(const std::string& imgName) const{
    imshow(imgName, image);
}

Mat* ImageProcessor::getImage(){
    return &image;
}

Mat *ImageProcessor::getGrayScale(){
    return &grayscale;
}

void ImageProcessor::resize(int height) {
    double scale = double(height) / image.rows;
    Size size = Size(image.cols * scale, height);
    cv::resize(image, image, size);
    cvtColor(image, grayscale, COLOR_BGR2GRAY);
}

