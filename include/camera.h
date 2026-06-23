#pragma once

struct CameraData
{
    int width;
    int height;

    float* image_gpu;

    float* gaussian_gpu;

    float* dog_gpu;

    CameraData()
    {
        width = 0;
        height = 0;

        image_gpu = nullptr;
        gaussian_gpu = nullptr;
        dog_gpu = nullptr;
    }
};
