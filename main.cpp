#include "ImageProcessor/ImageProcessor.h"
#include "MatchingAlgorithm/MatchingAlgorithm.h"
#include "FeatureDetector/MySIFT.h"
#include "Stitcher/Stitcher.h"
using namespace std;

int main() {
    string img_list[] = {"../image/3.jpg", "../image/1.jpg", "../image/2.jpg"};
    vector<ImageProcessor*> images;
    for(auto& img : img_list){
        auto* img_processor = new ImageProcessor(img);
        images.emplace_back(img_processor);
    }
    MatchingAlgorithm match(images);
    match.train(0);
    Stitcher stitcher(match);
    stitcher.train(false);
    auto res = stitcher.getResult();
    cv::imwrite("./2.png", res);
    return 0;
}