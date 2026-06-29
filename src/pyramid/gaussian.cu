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
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    int index = y * width + x;

    output[index] = input[index];
}

void gaussianBlur(
    const float* input_image,
    float* output_image,
    int width,
    int height)
{
    dim3 blockSize(16,16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    gaussianKernel<<<gridSize, blockSize>>>(
        input_image,
        output_image,
        width,
        height
    );

    cudaDeviceSynchronize();

    std::cout << "Gaussian Copy Kernel Executed" << std::endl;
}
