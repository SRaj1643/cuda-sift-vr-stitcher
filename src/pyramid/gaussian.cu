#include "gaussian.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#define MAX_KERNEL_SIZE 31

__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__global__
void gaussianKernel(
    const float* input,
    float* output,
    int width,
    int height,
    int kernelSize,
    int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.0f;

    for(int ky = -radius; ky <= radius; ky++)
    {
        for(int kx = -radius; kx <= radius; kx++)
        {
            int xx = min(max(x + kx, 0), width - 1);
            int yy = min(max(y + ky, 0), height - 1);

            float pixel = input[yy * width + xx];

            float weight =
                d_kernel[(ky + radius) * kernelSize + (kx + radius)];

            sum += pixel * weight;
        }
    }

    output[y * width + x] = sum;
}

void gaussianBlurCUDA(
    const float* input,
    float* output,
    int width,
    int height,
    float sigma)
{
    int radius = static_cast<int>(ceil(3.0f * sigma));

    int kernelSize = 2 * radius + 1;

    std::vector<float> kernel(kernelSize * kernelSize);

    float sum = 0.0f;

    for(int y = -radius; y <= radius; y++)
    {
        for(int x = -radius; x <= radius; x++)
        {
            float value =
                exp(-(x * x + y * y) /
                    (2.0f * sigma * sigma));

            kernel[(y + radius) * kernelSize +
                   (x + radius)] = value;

            sum += value;
        }
    }

    for(float& value : kernel)
        value /= sum;

    cudaMemcpyToSymbol(
        d_kernel,
        kernel.data(),
        kernel.size() * sizeof(float));

    dim3 block(16,16);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    gaussianKernel<<<grid,block>>>(
        input,
        output,
        width,
        height,
        kernelSize,
        radius);

    cudaDeviceSynchronize();
}
