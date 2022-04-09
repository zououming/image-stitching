//
// Created by zououming on 2022/1/23.
//

#include "MatchingAlgorithm.h"
using namespace cv;
using namespace std;

void MatchingAlgorithm::showFeaturePoints(const vector<ImageProcessor*>& imageList, const vector<vector<DMatch>>& matchList) {
    cv::Mat imgMatches;
    for(int i = 1; i < imageList.size(); i++) {
        drawMatches(imageList[i - 1]->getImage(), imageList[i - 1]->getKeyPoint(),
                    imageList[i]->getImage(), imageList[i]->getKeyPoint(),
                    matchList[i - 1], imgMatches);//进行绘制
        imshow("匹配图: " + imageList[i-1]->getPath() + " to " + imageList[i]->getPath(), imgMatches);
        cv::waitKey(0);
    }
}

std::vector<cv::DMatch> MatchingAlgorithm::getMatchPoints(int i) const {
    return matchPointsList[i];
}

void MatchingAlgorithm::train(bool show) {
    sortImage(images);
    featurePointsMatch();
    if(show) showFeaturePoints(images, matchPointsList);
}

void MatchingAlgorithm::featurePointsMatch() {
    matchPointsList.clear();
    for(int i = 0; i < images.size() - 1; i++)
        matchPointsList.emplace_back(twoImageMatch(i, i + 1));
}

std::vector<cv::DMatch> MatchingAlgorithm::twoImageMatch(int i, int j) {
    std::vector<std::vector<DMatch>> matchRes;
    std::vector<Mat> train_desc(1, images[j]->getDescriptor());
    matcher.clear();
    matcher.add(train_desc);
    matcher.train();
    matcher.knnMatch(images[i]->getDescriptor(), matchRes, 2);

    std::vector<cv::DMatch> matchPoints; //优秀匹配点
    double threshold = precision / 10.0;
    for(auto& match : matchRes)
        if(match[0].distance < match[1].distance * threshold)
            matchPoints.push_back(match[0]);
    return matchPoints;
}

void MatchingAlgorithm::findMaxIJ(const vector<vector<int>>& matrix, int& resI, int& resJ){
    int max_i = 0, max_j = 0, max_match = 0;
    for(int i = 0; i < matrix.size(); i++)
        for(int j = 0; j < matrix[0].size(); j++)
            if(matrix[i][j] > max_match){
                max_match = matrix[i][j];
                max_i = i;
                max_j = j;
            }
    if(max_i > max_j) swap(max_i, max_j);
    resI = max_i;
    resJ = max_j;
}

vector<vector<bool>> MatchingAlgorithm::kruskal(vector<vector<int>> graph) {
    vector<vector<bool>> res(graph.size(), vector<bool>(graph[0].size()));
    set<int> visited_set;
    int i, j;
    while(visited_set.size() < graph.size()) {
        findMaxIJ(graph, i, j);
        if(visited_set.find(i) == visited_set.end()){
            visited_set.insert(i);
            res[j][i] = true;
        }
        if(visited_set.find(j) == visited_set.end()){
            visited_set.insert(j);
            res[i][j] = true;
        }
        graph[i][j] = graph[j][i] = 0;
    }
    return res;
}

int MatchingAlgorithm::findCenter(const std::vector<std::vector<int>> &graph) {
    int max_line = 0, max_match_points = 0;
    for(int i = 0; i < graph.size(); i++) {
        int count = 0;
        for (auto& points : graph[i])
            count += points;
        if(count > max_match_points){
            max_line = i;
            max_match_points = count;
        }
    }
    return max_line;
}

int findMaxJ(const std::vector<std::vector<int>> &graph, int i){
    int max_pos = -1, max = 0;
    for(int j = 0; j < graph[i].size(); j++){
        if(graph[i][j] > 10 && graph[i][j] > max) {
            max_pos = j;
            max = graph[i][j];
        }
    }
    return max_pos;
}

void MatchingAlgorithm::sortImage(std::vector<ImageProcessor*> &imageList) {
    vector<vector<vector<DMatch>>> match_points_matrix(imageList.size(), vector<vector<DMatch>>(imageList.size()));
    vector<vector<int>> match_graph(imageList.size(), vector<int>(imageList.size()));
    int i, j;

    for(i = 0; i < imageList.size(); i++) {
        for (j = i + 1; j < imageList.size(); j++) {
            auto matchPoints = twoImageMatch(i, j);
            match_graph[i][j] = matchPoints.size();
            match_graph[j][i] = matchPoints.size();
        }
    }

    int offset = 0;
    for(i = 0; i < match_graph.size(); i++) {
        int count = 0;
        for (j = 0; j < match_graph[i].size(); j++){
            cout << match_graph[i][j] << " ";
            count += match_graph[i][j];
        }
        if(count < 10){
            cout << "移除无关图像: " << imageList[i]->getPath();
            offset++;
        }
        cout<<endl;
    }

    vector<ImageProcessor*> temp_images;
    midIndex = findCenter(match_graph);
    temp_images.emplace_back(imageList[midIndex]);
    vector<bool> visited(imageList.size());
    visited[midIndex] = true;
    int pos = midIndex, visit_count = 0;
    while(temp_images.size() + offset < imageList.size()){
        i = findMaxJ(match_graph, pos);
        if(i == -1) {
            visit_count++;
            pos = midIndex;
        } else if(visited[i]) {
            match_graph[pos][i] = match_graph[i][pos] = 0;
            continue;
        }else{
            visited[i] = true;
            match_graph[pos][i] = match_graph[i][pos] = 0;
            if (visit_count == 0)
                temp_images.emplace_back(imageList[i]);
            else
                temp_images.insert(temp_images.begin(), imageList[i]);
            pos = i;
        }
    }
    imageList = temp_images;
}

bool cmp(ImageProcessor* img1, ImageProcessor* img2){
    auto corner1 = img1->getCorner(), corner2 = img2->getCorner();
    if(corner1[2].x == corner2[2].x)
        return corner1[0].x < corner2[0].x;
    return corner1[2].x < corner2[2].x;
}

int MatchingAlgorithm::sortImageByPosition(vector<ImageProcessor*> &imageList) {
    cout<<"重新排序后顺序"<<endl;
    sort(imageList.begin(), imageList.end(), cmp);
    for(auto& image : imageList)
        image->imageShow(image->getPath(), 0);
}
