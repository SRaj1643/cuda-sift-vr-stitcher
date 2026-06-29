#pragma once

void gaussianBlurCUDA(
    const float* input,
    float* output,
    int width,
    int height,
    float sigma
);
