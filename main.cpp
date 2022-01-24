#include "ImageProcessor/ImageProcessor.h"
#include "MatchingAlgorithm/MySURF.h"
int main() {
    ImageProcessor ip("../image/home1.jpg");
    ImageProcessor ip2("../image/home2.jpg");
    ip.resize(500);
    ip2.resize(500);
    MySURF surf(&ip, &ip2, 2000, precision{HIGH});
    surf.train(true);
    return 0;
}
