cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME ImageStitching)

project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)

add_executable(ImageStitching main.cpp
        ImageProcessor/ImageProcessor.cpp
        ImageProcessor/ImageProcessor.h
        MatchingAlgorithm/MySURF.cpp
        MatchingAlgorithm/MySURF.h Stitcher/Stitcher.cpp Stitcher/Stitcher.h MatchingAlgorithm/MatchingAlgorithm.cpp MatchingAlgorithm/MatchingAlgorithm.h)
find_package(OpenCV REQUIRED)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
