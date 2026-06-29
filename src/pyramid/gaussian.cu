#include "gaussian.h"

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__
void gaussianKernel(
    const float* input,
    float* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    int idx = y * width + x;

    output[idx] = input[idx];
}

void gaussianBlurCUDA(
    const float* input,
    float* output,
    int width,
    int height,
    float sigma)
{
    dim3 blockSize(16,16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    gaussianKernel<<<gridSize,blockSize>>>(
        input,
        output,
        width,
        height
    );

    cudaDeviceSynchronize();
}
