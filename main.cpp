#include "ImageProcessor/ImageProcessor.h"
#include "MatchingAlgorithm/MatchingAlgorithm.h"
#include "FeatureDetector/MySIFT.h"
#include "Stitcher/Stitcher.h"
using namespace std;

int main() {
//    string img_list[] = {"../image/11.jpg", "../image/22.jpg", "../image/33.jpg", "../image/1.jpg", "../image/44.jpg"};
    string img_list[] = {"../image/1.jpg", "../image/2.jpg", "../image/3.jpg"};
    vector<ImageProcessor*> images;
    for(auto& img : img_list){
        auto* img_processor = new ImageProcessor(img, 500);
        images.emplace_back(img_processor);
    }
    MatchingAlgorithm match(images);
    match.train(1);
    Stitcher stitcher(match, 0);
    stitcher.train(false);
    auto res = stitcher.getResult();
    cv::imshow("fres", res);
    cv::imwrite("./233.png", res);
    cv::waitKey(0);
    return 0;
}