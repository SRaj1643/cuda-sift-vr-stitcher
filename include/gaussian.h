#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#define NUM_OCTAVES 4
#define NUM_SCALES 8

// Same sigma values as CPU version
extern const float SIGMAS[NUM_SCALES];

// ----------------------------------------------------
// CPU Reference
// ----------------------------------------------------
std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCPU(
    const cv::Mat& image
);

// ----------------------------------------------------
// CUDA Version
// ----------------------------------------------------
std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCUDA(
    const cv::Mat& image
);

// ----------------------------------------------------
// Save Pyramid Images
// ----------------------------------------------------
void saveGaussianPyramid(
    const std::vector<std::vector<cv::Mat>>& pyramid,
    const std::string& outputFolder
);

#endif
