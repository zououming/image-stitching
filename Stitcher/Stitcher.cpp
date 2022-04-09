//
// Created by zououming on 2022/1/23.
//

#include "Stitcher.h"
using namespace std;
using namespace cv::detail;

void Stitcher::train(const bool &show) {
    homographyCompute(0);
    for(int i = 0; i < algorithm->images.size(); i++)
        algorithm->images[i]->setCorner(cornersCompute(i));

    algorithm->sortImageByPosition(algorithm->images); //重新排序
    algorithm->featurePointsMatch();                      //重新匹配

    homographyCompute(1);
    for(int i = 0; i < algorithm->images.size(); i++)
        algorithm->images[i]->setCorner(cornersCompute(i));

    resultSize = resultSizeCompute();
    cout<<resultSize<<endl;

    for(int i = 0; i < algorithm->images.size(); i++) {
        transformImages.emplace_back(cv::Mat());
        transform(i, resultSize);
    }
//    ExposureAdjustment();

    stitchedImage = cv::Mat(resultSize, CV_8UC3);
    stitchedImage.setTo(0);
    for(int i = 0; i <algorithm->images.size(); i++)
        imageStitch(i, stitchedImage);
    for(int i = 0; i < algorithm->images.size() - 1 ; i++) {
        int j = i + 1;
//        for(int j = i + 1; j < algorithm->images.size(); j++){
            auto img1 = algorithm->images[i], img2 = algorithm->images[j];
            auto rec = overlappingAreaCompute(img1->getCorner(), img2->getCorner());
            auto m = stitchedImage.clone();
            imageFuse(i, j, rec[0], rec[1], 2);
            cv::rectangle(m, cv::Rect(rec[0], rec[1]), cv::Scalar(255,255,255));
            cv::imshow("res", stitchedImage);
            cv::waitKey(0);
//        }
    }
    cout<<midIndex<<endl;
    auto leftCorner = algorithm->images[0]->getCorner(),
        rightCorner = algorithm->images[algorithm->images.size()-1]->getCorner();
    cv::Point st(MAX(MAX(leftCorner[leftTop].x, leftCorner[leftBottom].x), 0), algorithm->images[midIndex]->getCorner()[leftTop].y);
    cv::Point ed(MIN(MIN(rightCorner[rightTop].x, rightCorner[rightBottom].x), resultSize.width), algorithm->images[midIndex]->getCorner()[rightBottom].y);
    stitchedImage = stitchedImage(cv::Rect(st, ed));
//    for(auto & image : algorithm->images){
//        auto corner = image->getCorner();
//        for(int j = 0; j < 4; j++) {
//            optimizeSeam(corner[j], corner[(j + 1) % 4], stitchedImage);
//            cv::imshow(image->getPath(), stitchedImage);
//            cv::waitKey(0);
//        }
//        cv::destroyWindow(image->getPath());
//    }
}

void Stitcher::transform(const int& i, const cv::Size& resultSize) {
    cv::warpPerspective(algorithm->images[i]->getImage(), transformImages[i], homographyMats[i], resultSize);
}

std::vector<cv::Point> Stitcher::cornersCompute(const int& index) {
    std::vector<cv::Point> imageCorners;
    cv::Mat image = algorithm->images[index]->getImage();
    double rows = image.rows, cols = image.cols;
    const vector<vector<double>> cornerVectors = {{0, 0, 1}, {0, rows, 1},
                                            {cols, 0, 1}, {cols, rows,1}};
    cout<<algorithm->images[index]->getPath() <<" corners: ";
    cv::Mat_<double> transPoint, vec;
    for(auto& v : cornerVectors){
        vec = cv::Mat(v);
        transPoint = homographyMats[index] * vec;
//        transPoint(0, 0) = max(transPoint(0, 0), .0);
//        transPoint(0, 1) = max(transPoint(0, 1), .0);
        imageCorners.emplace_back(transPoint(0, 0)/ transPoint(0, 2),
                                  transPoint(0, 1)/ transPoint(0, 2));
    }
    for(auto& corner : imageCorners)
        cout << corner << " ";
    cout<<endl;
    return imageCorners;
}

void Stitcher::imageStitch(int i, cv::Mat& resultImg) {
    auto corner = algorithm->images[i]->getCorner();
    auto leftTopCorner = corner[leftTop];
//    corner[rightBottom].x = min(resultSize.width, corner[rightBottom].x);
//    corner[rightBottom].y = min(resultSize.height, corner[rightBottom].y);
//    auto roi = transformImages[i](cv::Rect(corner[leftTop], corner[rightBottom]));]
    auto bounding = boundingRectangleCompute(corner);
    auto roi = transformImages[i](cv::Rect(bounding[0], bounding[1]));
//    cv::imshow("roi", roi);
//    cv::waitKey(0);
//    roi.copyTo(resultImg(cv::Rect(leftTopCorner.x, leftTopCorner.y, roi.cols, roi.rows)));
    roi.copyTo(resultImg(cv::Rect(bounding[0].x, bounding[0].y, roi.cols, roi.rows)));
    cv::imshow("res", resultImg);
    cv::waitKey(0);
}

int8_t Stitcher::homographyCompute(const bool& shift) {
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

    if(!shift) return 0;

    //平移
    const vector<vector<double>> corner = {{0, 0, 1}, {0, double(algorithm->images[0]->getImage().rows), 1},
                                           {double(algorithm->images[0]->getImage().cols), 0, 1}}; //左上, 左下, 右上
    Mat_<double> left_top = homographyMats[0] * Mat(corner[0]);
    Mat_<double> left_bottom = homographyMats[0] * Mat(corner[1]);
    Mat_<double> right_top = homographyMats[0] * Mat(corner[2]);

    double top_x = left_top.at<double>(0, 0), bottom_x = left_bottom.at<double>(0, 0);
    double left_y = left_top.at<double>(0, 1), right_y = right_top.at<double>(0, 1);

       Mat shiftMat = Mat::eye(3, 3, CV_64F);
    if(top_x < 0 || bottom_x < 0)
        shiftMat.at<double>(0, 2) = - double(MIN(top_x, bottom_x));
    if(left_y < 0 || right_y < 0)
        shiftMat.at<double>(1, 2) = - double(MIN(left_y, right_y));
    for(auto& homography : homographyMats)
        homography = shiftMat * homography;

    return 1;
}

cv::Size Stitcher::resultSizeCompute() {
    int max_x = 0, max_y = 0;
    for(auto& img : algorithm->images){
        auto corner = img->getCorner();
        max_x = MAX(MAX(corner[rightTop].x, corner[rightBottom].x), max_x);
        max_y = MAX(MAX(corner[leftBottom].y, corner[rightBottom].y), max_y);
    }
    return {max_x, max_y};
}

void Stitcher::imageFuse(const int& index1, const int& index2, cv::Point p1, cv::Point p2, const int& expansion) {
    //曝光补偿
//    vector<vector<uchar>> BGR1(3), BGR2(3);
//    for(int i = p1.y; i < p2.y; i++) {
//        auto *pt1 = transformImages[index1].ptr<uchar>(i);
//        auto *pt2 = transformImages[index2].ptr<uchar>(i);
//        for (int j = p1.x; j < p2.x; j++) {
//            if(pt1[j * 3] == 0 && pt1[j * 3 + 1] == 0 && pt1[j * 3 +2] == 0
//            || pt2[j * 3] == 0 && pt2[j * 3 + 1] == 0 && pt2[j * 3 +2] == 0) continue;
//
//            for(int channel = 0; channel < 3; channel++) {
//                BGR1[channel].emplace_back(pt1[j * 3 + channel]);
//                BGR2[channel].emplace_back(pt2[j * 3 + channel]);
//            }
//        }
//    }
//    cv::Vec4f lines;
//    vector<vector<double>> parameter;
//    for(int i = 0; i < 3; i++) {
//        parameter.emplace_back(fitLine(BGR2[i], BGR1[i]));
//    }
//    auto img2 = algorithm->images[index2]->getImage();
//    for(int i = 0; i < img2.rows; i++) {
//        auto *pt2 = img2.ptr<uchar>(i);
//        for (int j = 0; j < img2.cols; j++) {
//            if(pt2[j * 3] == 0 && pt2[j * 3 + 1] == 0 && pt2[j * 3 +2] == 0) continue;
//            for(int channel = 0; channel < 3; channel++) {
//                pt2[j * 3 + channel] = uchar(pt2[j * 3 + channel] * parameter[channel][0] + parameter[channel][1]);
//            }
//        }
//    }
//    transform(index2, resultSize);
//    cv::imshow("1", transformImages[index2]);
//    algorithm->images[index2]->imageShow("2", 0);

    //过渡处理
    p1.x = MAX(0, p1.x - expansion);
    p1.y = MAX(0, p1.y - expansion);
    p2.x = MIN(resultSize.width, p2.x + expansion);
    p2.y = MIN(resultSize.height, p2.y + expansion);
    double processWidth = p2.x - p1.x;
    double alpha = 1;
    for (int i = p1.y; i < p2.y; i++){
        auto* p = transformImages[index1].ptr<uchar>(i);
        auto* t = transformImages[index2].ptr<uchar>(i);
        auto* d = stitchedImage.ptr<uchar>(i);
        for (int j = p1.x; j < p2.x; j++){
            bool pixel1 = p[j * 3] == 0 && p[j * 3 + 1] == 0 && p[j * 3 + 2] == 0;  //img1为黑点
            bool pixel2 = t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0;  //img2为黑点
            bool pixelRes = d[j * 3] != 0 || d[j * 3 + 1] != 0 || d[j * 3 + 2] != 0;//Res不为黑点
            if (pixel1 && pixel2 && pixelRes) continue;
            if (pixel2) alpha = 1;
            else if(pixel1) alpha = 0;
            else if(optMethod == liner) alpha = (processWidth - (j - p1.x)) / processWidth;
            else if(optMethod == easeInOut) alpha = easeInOutOpt((j - p1.x) / processWidth);

            d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
            d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
            d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
        }
    }
}

double Stitcher::easeInOutOpt(double x) {
    return x < 0.5? 1 - 2*x*x : 2*(1-x)*(1-x);
}

cv::Mat Stitcher::getResult() {
    return stitchedImage;
}

vector<cv::Point> Stitcher::overlappingAreaCompute(vector<cv::Point> corners1, vector<cv::Point> corners2) {
    vector<cv::Point> result;
    auto ret1 = boundingRectangleCompute(corners1), ret2 = boundingRectangleCompute(corners2);
    auto leftTop1 = ret1[0], rightBottom1 = ret1[1], leftTop2 = ret2[0], rightBottom2 = ret2[1];

    if(leftTop2.y < rightBottom1.y && leftTop2.y > leftTop1.y)
        return {leftTop2, rightBottom1};
    cv::Point st(leftTop2.x, leftTop1.y);
    cv::Point ed(rightBottom1.x, rightBottom2.y);
    return {st, ed};
}

std::vector<cv::Point> Stitcher::boundingRectangleCompute(const vector<cv::Point> &corner) {
    cv::Point leftTopPoint(MAX(MIN(corner[leftTop].x, corner[leftBottom].x),0), MAX(MIN(corner[leftTop].y, corner[rightTop].y), 0));
    cv::Point rightBottomPoint(MAX(corner[rightTop].x, corner[rightBottom].x), MAX(corner[leftBottom].y, corner[rightBottom].y));
    return {leftTopPoint, rightBottomPoint};
}

void Stitcher::ExposureAdjustment() {
    cv::Ptr<ExposureCompensator> compensator =
            ExposureCompensator::createDefault(ExposureCompensator::GAIN);
    vector<cv::Point> corners;
    vector<cv::UMat> transUMat, transMask;
    int i = 0;
    for (auto &img : algorithm->images) {
        cv::UMat mat;
        corners.emplace_back(img->getCorner()[0]);
        transformImages[i++].copyTo(mat);
        transUMat.emplace_back(mat);
        transMask.emplace_back(mat);
    }
    transformImages.clear();
    compensator->feed(corners, transUMat, transMask);    //得到曝光补偿器
    for (i = 0; i < algorithm->images.size(); ++i){   //应用曝光补偿器，对图像进行曝光补偿
        compensator->apply(i, corners[i], transUMat[i], transMask[i]);
        transformImages.emplace_back(transMask[i].getMat(cv::ACCESS_WRITE));
    }
}

std::vector<double> Stitcher::fitLine(const vector<uchar> &x, const std::vector<uchar> &y) {
    double w = 0, b = 0, ave_x = 0, sum_x = 0, sum_square_x = 0;
    for(auto& xx : x) {
        sum_x += xx;
        sum_square_x += xx * xx;
    }
    ave_x = sum_x / x.size();
    for(int i = 0; i < x.size(); i++)
        w += y[i] * (x[i] - ave_x);
    w /= (sum_square_x - sum_x*sum_x / x.size());
    for(int i = 0; i < x.size(); i++)
        b += y[i] - w * x[i];
    b /= x.size();

    return {w, b};
}

void Stitcher::optimizeSeam(const cv::Point &p1, const cv::Point &p2, cv::Mat &img) {
    double k, b;
    k = double(p2.y - p1.y) / (p2.x - p1.x);
    b = p1.y - (p1.x * k);
    for(int x = p1.x; x < p2.x; x++){
        int count[3] = {0}, sum[3] = {0};
        int y = k * x + b;
        if(y < 0 || y > img.rows) continue;
        for(int i = -1; i < 2; i++) {
            if(y + i < 0 || y + i > img.rows) continue;
            auto pt = img.ptr(y + i);
            for (int j = -1; j < 2; j++) {
                int pixel_x = x + i;
                if (pixel_x < 0 || pixel_x > img.cols) continue;
                if (int(pixel_x * k + b) == int(y + i)) continue;
                if (pt[pixel_x * 3] == 0 && pt[pixel_x * 3 + 1] == 0 && pt[pixel_x * 3 + 2] == 0) continue;
                for(int channel = 0; channel < 3; channel++){
                    sum[channel] += pt[pixel_x * 3 + channel];
                    count[channel]++;
                }
            }
        }
        auto pt = img.ptr(y);
        for(int channel = 0; channel < 3; channel++)
            pt[x * 3 + channel] = int(sum[channel] / count[channel]);
        if(y - 1 > 0){
            pt = img.ptr(y - 1);
            for(int channel = 0; channel < 3; channel++)
                pt[x * 3 + channel] = int(sum[channel] / count[channel]);
        }
        if(y + 1 < img.rows){
            pt = img.ptr(y + 1);
            for(int channel = 0; channel < 3; channel++)
                pt[x * 3 + channel] = int(sum[channel] / count[channel]);
        }
    }
}
