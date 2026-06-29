#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <opencv2/opencv.hpp>
#include <vector>

#define NUM_OCTAVES 4
#define NUM_SCALES 8

// Same sigma values as CPU version
extern const float SIGMAS[NUM_SCALES];

// CPU reference function
std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCPU(
    const cv::Mat& image
);

// CUDA version
std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCUDA(
    const cv::Mat& image
);

#endif
