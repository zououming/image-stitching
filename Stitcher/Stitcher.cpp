//
// Created by zououming on 2022/1/23.
//

#include "Stitcher.h"
using namespace std;

void Stitcher::train(const bool &show) {
    homographyCompute();
    transform();
    imageStitch();
    for(int i = 0; i < algorithm->images.size() - 1; i++)
        optimizeSeam(i, i + 1);
}

void Stitcher::transform() {
    int size = algorithm->images.size();
    for(int i = 0; i < size; i++)
        corners.emplace_back(cornersCompute(i));
    resultSizeCompute(algorithm->images[midIndex]->getImage().rows);
    for(int i = 0; i < size; i++) {
        cv::Mat transformImage;
        cv::warpPerspective(algorithm->images[i]->getImage(), transformImage, homographyMats[i], resultSize);
        transformImages.emplace_back(transformImage);
    }
}

std::vector<cv::Point> Stitcher::cornersCompute(const int& index) {
    std::vector<cv::Point> imageCorners;
    cv::Mat image = algorithm->images[index]->getImage();
    double rows = image.rows, cols = image.cols;
    const vector<vector<double>> cornerVectors = {{0, 0, 1}, {0, rows, 1},
                                            {cols, 0, 1}, {cols, rows,1}};
    cout<<"image "<< index <<" corners: ";
    cv::Mat_<double> transPoint, vec;
    for(auto& v : cornerVectors){
        vec = cv::Mat(v);
        transPoint = homographyMats[index] * vec;
        transPoint(0, 0) = max(transPoint(0, 0), .0);
        transPoint(0, 1) = max(transPoint(0, 1), .0);
        imageCorners.emplace_back(transPoint(0, 0)/ transPoint(0, 2),
                                  transPoint(0, 1)/ transPoint(0, 2));
    }
    for(auto& corner : imageCorners)
        cout << corner << " ";
    cout<<endl;
    return imageCorners;
}

void Stitcher::imageStitch() {
    stitchedImage = cv::Mat(resultSize, CV_8UC3);
    stitchedImage.setTo(0);
    int i = 0;
    for(auto& image : transformImages) {
        auto corner = corners[i++];
        auto leftTopCorner = corner[leftTop];
        corner[rightBottom].x = min(resultSize.width, corner[rightBottom].x);
        corner[rightBottom].y = min(resultSize.height, corner[rightBottom].y);
        auto roi = image(cv::Rect(corner[leftTop], corner[rightBottom]));
        cv::imshow("roi", roi);
        cv::waitKey(0);
        roi.copyTo(stitchedImage(cv::Rect(leftTopCorner.x, leftTopCorner.y, roi.cols, roi.rows)));
        cv::imshow("res", stitchedImage);
        cv::waitKey(0);
    }
}

int8_t Stitcher::homographyCompute() {
    using namespace cv;
    int8_t size = algorithm->images.size();
    homographyMats = std::vector<Mat>(size);
    midIndex = size%2 == 0? size / 2 - 1 : size / 2;
    homographyMats[midIndex] = Mat::eye(3, 3, CV_64F);

    for(int i = midIndex - 1; i >= 0; i--) {  //左半边
        vector<Point2f> imagePoints1, imagePoints2;
        vector<KeyPoint> keyPoint1, keyPoint2;
        keyPoint1 = algorithm->images[i + 1]->getKeyPoint();
        keyPoint2 = algorithm->images[i]->getKeyPoint();
        for (auto &match : algorithm->getMatchPoints(i)) {
            imagePoints1.push_back(keyPoint1[match.trainIdx].pt);
            imagePoints2.push_back(keyPoint2[match.queryIdx].pt);
        }
        auto homo = findHomography(imagePoints2, imagePoints1);
        homographyMats[i] = homo * homographyMats[i + 1];
    }

    for(int i = midIndex + 1; i < size; i++) {  //右半边
        vector<Point2f> imagePoints1, imagePoints2;
        vector<KeyPoint> keyPoint1, keyPoint2;
        keyPoint1 = algorithm->images[i]->getKeyPoint();
        keyPoint2 = algorithm->images[i - 1]->getKeyPoint();
        for (auto &match : algorithm->getMatchPoints(i - 1)) {
            imagePoints1.push_back(keyPoint1[match.trainIdx].pt);
            imagePoints2.push_back(keyPoint2[match.queryIdx].pt);
        }
        auto homo = findHomography(imagePoints1, imagePoints2);
        homographyMats[i] = homographyMats[i - 1] * homo;
    }

    const vector<vector<double>> cornerVectors = {{0, 0, 1}, {0, double(algorithm->images[0]->getImage().rows), 1}};
    Mat_<double> left_top = homographyMats[0] * Mat(cornerVectors[0]);
    Mat_<double> left_bottom = homographyMats[0] * Mat(cornerVectors[1]);
    double top_x = left_top.at<double>(0, 0), bottom_x = left_bottom.at<double>(0, 0);

    Mat shiftMat = Mat::eye(3, 3, CV_64F);
    if(top_x < 0 || bottom_x < 0)
        shiftMat.at<double>(0, 2) = - double(min(top_x, bottom_x));
    for(auto& homography : homographyMats)
        homography = shiftMat * homography;

    return 1;
}

void Stitcher::resultSizeCompute(int midHeight) {
    auto last_corner = corners[corners.size() - 1];
    int max_x = min(last_corner[rightTop].x, last_corner[rightBottom].x);
    resultSize = cv::Size(max_x, midHeight);
    cout<<"result size"<<resultSize<<endl;
}

void Stitcher::optimizeSeam(int i, int j) {
    auto img1 = transformImages[i], img2 = transformImages[j];

    int start = MIN(corners[j][leftTop].x, corners[j][leftBottom].x);
    int end = MAX(corners[i][rightTop].x, corners[i][rightBottom].x);
    double processWidth = end - start;//重叠区域的宽度
    double alpha = 1;//img1中像素的权重
    for (int i = 0; i < resultSize.height; i++)
    {
        uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
        uchar* t = img2.ptr<uchar>(i);
        uchar* d = stitchedImage.ptr<uchar>(i);
        for (int j = start; j < end; j++)
        {
            //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
            if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0) alpha = 1;
            else if(optMethod == liner) alpha = (processWidth - (j - start)) / processWidth;
            else if(optMethod == easeInOut) alpha = easeInOutOpt((j - start) / processWidth);

            d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
            d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
            d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
        }
    }
}

double Stitcher::easeInOutOpt(double x) {
    if(x < 0.5)
        return 1 - 2*x*x;
    else
        return 2*(1-x)*(1-x);
}

cv::Mat Stitcher::getResult() {
    return stitchedImage;
}
