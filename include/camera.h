
#pragma once

struct CameraData
{
    int width;
    int height;

    // Original image
    float* image_gpu;

    // Gaussian Pyramid
    float* gaussian_gpu;

    // DoG Pyramid
    float* dog_gpu;
};
