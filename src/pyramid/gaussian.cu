#include "gaussian.h"

#include <cuda_runtime.h>
#include <iostream>

__global__
void gaussianKernel(
    const float* input,
    float* output,
    int width,
    int height)
{
    // Empty kernel for now
}

void gaussianBlur(
    const float* input_image,
    float* output_image,
    int width,
    int height)
{
    std::cout << "Gaussian Blur Module Ready" << std::endl;
}
