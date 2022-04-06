//
// Created by zououming on 2022/1/23.
//

#include "MatchingAlgorithm.h"
using namespace cv;
using namespace std;

void MatchingAlgorithm::showFeaturePoints() {
    cv::Mat imgMatches;
    for(int i = 1; i < images.size(); i++) {
        drawMatches(images[i - 1]->getImage(), images[i - 1]->getKeyPoint(),
                    images[i]->getImage(), images[i]->getKeyPoint(),
                    matchPointsList[i - 1], imgMatches);//进行绘制
        imshow("匹配图: " + images[i-1]->getPath() + " to " + images[i]->getPath(), imgMatches);
        cv::waitKey(0);
    }
}

std::vector<cv::DMatch> MatchingAlgorithm::getMatchPoints(int i) const {
    return matchPointsList[i];
}

void MatchingAlgorithm::train(bool show) {
    featurePointsMatch();
    if(show) showFeaturePoints();
}

void MatchingAlgorithm::featurePointsMatch() {
    vector<vector<int>> match_graph(images.size(), vector<int>(images.size()));
    vector<vector<vector<DMatch>>> match_points_matrix(images.size(), vector<vector<DMatch>>(images.size()));
    int i, j;
    for(i = 0; i < images.size(); i++)
        for(j = i + 1; j < images.size(); j++){
            auto matchPoints = twoImageMatch(i, j);
            match_points_matrix[i][j] = matchPoints;
            match_graph[i][j] = matchPoints.size();
            match_graph[j][i] = matchPoints.size();
        }
    auto mst = kruskal(match_graph);
    for(i = 0; i < mst.size(); i++) {
        int count = 0;
        for(j = 0; j < mst[i].size(); j++)
            count += mst[i][j];
        if(count == 1) break;   //找到起点
    }
    vector<ImageProcessor*> temp_images;
    temp_images.emplace_back(images[i]);
    while(temp_images.size() < images.size()){
        for(j = 0; j < images.size(); j++)
            if (mst[i][j]) {
                temp_images.emplace_back(images[j]);
                mst[i][j] = mst[j][i] = false;
                i = j;
                break;
            }
    }
    images = temp_images;
    for(i = 0; i < images.size() - 1; i++)
        matchPointsList.emplace_back(twoImageMatch(i, i + 1));
}

std::vector<cv::DMatch> MatchingAlgorithm::twoImageMatch(int i, int j) {
    FlannBasedMatcher matcher;
    std::vector<std::vector<DMatch>> matchRes;
    std::vector<Mat> train_desc(1, images[j]->getDescriptor());

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
