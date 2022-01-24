//
// Created by zououming on 2022/1/21.
//

#ifndef IMAGESTITCHING_IMAGEPROCESSOR_H
#define IMAGESTITCHING_IMAGEPROCESSOR_H
#include <opencv2/opencv.hpp>

class ImageProcessor{
public:
    explicit ImageProcessor(const std::string& imgPath);
    void imageShow(const std::string& imgName) const;
    void resize(int height = 800);
    cv::Mat* getImage();
    cv::Mat* getGrayScale();
private:
    cv::Mat image;
    cv::Mat grayscale;
};


#endif //IMAGESTITCHING_IMAGEPROCESSOR_H
