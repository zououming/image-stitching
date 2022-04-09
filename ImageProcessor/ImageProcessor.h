//
// Created by zououming on 2022/1/21.
//

#ifndef IMAGESTITCHING_IMAGEPROCESSOR_H
#define IMAGESTITCHING_IMAGEPROCESSOR_H
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

class ImageProcessor{
    std::string path;
    cv::Mat image, transImage;
    cv::Mat grayScale;
    std::vector<cv::KeyPoint> keyPoint;
    cv::Mat descriptor;
    std::vector<cv::Point> corner;
public:
    explicit ImageProcessor(const std::string& imgPath, const int& height = 500);
    void imageShow(const std::string& imgName, const int& time = 0);
    void resize(int height = 800);
    cv::Mat getImage();
    void setCorner(std::vector<cv::Point>);
    std::vector<cv::Point> getCorner();
    std::vector<cv::KeyPoint> getKeyPoint();
    cv::Mat getDescriptor();
    std::string getPath();
    enum {leftTop, leftBottom, rightTop, rightBottom};
};


#endif //IMAGESTITCHING_IMAGEPROCESSOR_H
