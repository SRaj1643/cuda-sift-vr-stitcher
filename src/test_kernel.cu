#include "../include/test_kernel.h"

#include <cuda_runtime.h>

__global__
void multiplyByTwoKernel(
    float* data,
    int size)
{
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(idx < size)
    {
        data[idx] *= 2.0f;
    }
}

void launchMultiplyByTwo(
    float* gpu_data,
    int size)
{
    int threads = 256;

    int blocks =
        (size + threads - 1) / threads;

    multiplyByTwoKernel<<<blocks, threads>>>(
        gpu_data,
        size);

    cudaDeviceSynchronize();
}
