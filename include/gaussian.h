#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

//--------------------------------------------------
// Gaussian Pyramid Configuration
//--------------------------------------------------

constexpr int NUM_OCTAVES = 4;
constexpr int NUM_SCALES  = 8;

// Fixed sigma values (same as CPU implementation)
extern const float SIGMAS[NUM_SCALES];

//--------------------------------------------------
// CPU Reference
//--------------------------------------------------

std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCPU(
    const cv::Mat& image
);

//--------------------------------------------------
// CUDA Version
//--------------------------------------------------

std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCUDA(
    const cv::Mat& image
);

//--------------------------------------------------
// Utility
//--------------------------------------------------

void saveGaussianPyramid(
    const std::vector<std::vector<cv::Mat>>& pyramid,
    const std::string& outputFolder
);

#endif
