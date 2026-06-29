#pragma once

#include "config.h"

struct PyramidLevel
{
    float* d_image = nullptr;

    int width = 0;

    int height = 0;
};

struct GaussianPyramid
{
    PyramidLevel level
        [NUM_OCTAVES]
        [NUM_SCALES];
};
