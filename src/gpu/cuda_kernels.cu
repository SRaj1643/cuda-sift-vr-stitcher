#include "cuda_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

//==========================================================
// CONSTANT MEMORY
//==========================================================

__constant__ float d_kernel[MAX_KERNEL_SIZE];

__constant__ int d_kernel_radius;

//==========================================================
// CUDA ERROR CHECK
//==========================================================

#define CUDA_CHECK(call)                                      \
{                                                             \
    cudaError_t err = call;                                   \
    if(err != cudaSuccess)                                    \
    {                                                         \
        printf("CUDA Error: %s\n",                            \
               cudaGetErrorString(err));                      \
        return;                                               \
    }                                                         \
}

//==========================================================
// COMPUTE GAUSSIAN RADIUS
//==========================================================

int getKernelRadius(float sigma)
{
    int radius =
        static_cast<int>(
            ceil(3.0f * sigma)
        );

    if(radius > MAX_KERNEL_RADIUS)
        radius = MAX_KERNEL_RADIUS;

    return radius;
}

//==========================================================
// GENERATE 1D GAUSSIAN KERNEL
//==========================================================

void generateGaussianKernel(

    float* kernel,

    int radius,

    float sigma

)
{
    float sum = 0.0f;

    int kernelSize =
        2 * radius + 1;

    for(int i=-radius;i<=radius;i++)
    {
        float value =
            expf(

                -(float)(i*i)

                /

                (2.0f*sigma*sigma)

            );

        kernel[
            i+radius
        ] = value;

        sum += value;
    }

    for(int i=0;i<kernelSize;i++)
    {
        kernel[i] /= sum;
    }
}

//==========================================================
// COPY GAUSSIAN KERNEL TO CONSTANT MEMORY
//==========================================================

void uploadGaussianKernel(

    float sigma

)
{
    int radius =
        getKernelRadius(
            sigma
        );

    int kernelSize =
        2*radius+1;

    float h_kernel[
        MAX_KERNEL_SIZE
    ];

    generateGaussianKernel(

        h_kernel,

        radius,

        sigma

    );

    cudaMemcpyToSymbol(

        d_kernel,

        h_kernel,

        kernelSize *
        sizeof(float)

    );

    cudaMemcpyToSymbol(

        d_kernel_radius,

        &radius,

        sizeof(int)

    );
}
//==========================================================
// HORIZONTAL GAUSSIAN BLUR
//==========================================================

__global__
void gaussianBlurHorizontal(
    const float* src,
    float* dst,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.0f;

    // Read radius from constant memory
    int radius = d_kernel_radius;

    for (int k = -radius; k <= radius; k++)
    {
        int xx = x + k;

        // Clamp boundary handling
        if (xx < 0)
            xx = 0;

        if (xx >= width)
            xx = width - 1;

        float pixel =
            src[y * width + xx];

        float weight =
            d_kernel[k + radius];

        sum += pixel * weight;
    }

    dst[y * width + x] = sum;
}
//==========================================================
// VERTICAL GAUSSIAN BLUR
//==========================================================

__global__
void gaussianBlurVertical(
    const float* src,
    float* dst,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.0f;

    int radius = d_kernel_radius;

    for (int k = -radius; k <= radius; k++)
    {
        int yy = y + k;

        if (yy < 0)
            yy = 0;

        if (yy >= height)
            yy = height - 1;

        float pixel =
            src[yy * width + x];

        float weight =
            d_kernel[k + radius];

        sum += pixel * weight;
    }

    dst[y * width + x] = sum;
}


//==========================================================
// DOWNSAMPLE (FACTOR 2)
//==========================================================

__global__
void downsampleKernel(
    const float* src,
    float* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_width || y >= dst_height)
        return;

    int srcX = x * 2;
    int srcY = y * 2;

    float sum = 0.0f;
    int count = 0;

    for(int dy = 0; dy < 2; dy++)
    {
        for(int dx = 0; dx < 2; dx++)
        {
            int xx = srcX + dx;
            int yy = srcY + dy;

            if(xx < src_width && yy < src_height)
            {
                sum += src[yy * src_width + xx];
                count++;
            }
        }
    }

    if(count > 0)
    {
        dst[y * dst_width + x] =
            sum / static_cast<float>(count);
    }
}
//==========================================================
// CUDA GAUSSIAN BLUR WRAPPER
//==========================================================

cudaError_t cudaGaussianBlur(
    float* d_input,
    float* d_output,
    float* d_temp,
    int width,
    int height,
    float sigma
)
{
    uploadGaussianKernel(sigma);

    dim3 block(16,16);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    gaussianBlurHorizontal<<<grid, block>>>(
        d_input,
        d_temp,
        width,
        height
    );

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess)
        return err;

    gaussianBlurVertical<<<grid, block>>>(
        d_temp,
        d_output,
        width,
        height
    );

    err = cudaGetLastError();

    if(err != cudaSuccess)
        return err;

    err = cudaDeviceSynchronize();

    return err;
}

//==========================================================
// CUDA DOWNSAMPLE WRAPPER
//==========================================================

cudaError_t cudaDownsample(
    float* d_input,
    float* d_output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
)
{
    dim3 block(16,16);

    dim3 grid(
        (dst_width + block.x - 1) / block.x,
        (dst_height + block.y - 1) / block.y
    );

    downsampleKernel<<<grid, block>>>(
        d_input,
        d_output,
        src_width,
        src_height,
        dst_width,
        dst_height
    );

    cudaError_t err =
        cudaGetLastError();

    if(err != cudaSuccess)
        return err;

    return cudaDeviceSynchronize();
}
