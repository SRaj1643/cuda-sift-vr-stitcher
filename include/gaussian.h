#pragma once

#include <vector>

void buildGaussianPyramid(
    const float* input_image,
    int width,
    int height,
    int num_octaves,
    int num_scales,
    std::vector<float*>& gaussian_pyramid
);
