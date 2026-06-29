#include "gaussian_kernel.cuh"

#include <cuda_runtime.h>

__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__constant__ int d_kernelSize;

#define BLOCK_SIZE 16

//---------------------------------------------------
// Upload Gaussian Kernel
//---------------------------------------------------

void uploadGaussianKernel(
    const float* kernel,
    int kernelSize
)
{
    cudaMemcpyToSymbol(
        d_kernel,
        kernel,
        kernelSize * kernelSize * sizeof(float)
    );

    cudaMemcpyToSymbol(
        d_kernelSize,
        &kernelSize,
        sizeof(int)
    );
}

//---------------------------------------------------
// CUDA Kernel
//---------------------------------------------------

__global__

void gaussianKernel(

    const float* input,

    float* output,

    int width,

    int height

)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    float sum = 0.0f;

    int radius = d_kernelSize / 2;

    for(int ky=-radius; ky<=radius; ky++)
    {
        for(int kx=-radius; kx<=radius; kx++)
        {
            int xx = min(max(x+kx,0),width-1);

            int yy = min(max(y+ky,0),height-1);

            float pixel =
                input[yy*width+xx];

            float weight =
                d_kernel[(ky+radius)*d_kernelSize + (kx+radius)];

            sum += pixel*weight;
        }
    }

    output[y*width+x] = sum;
}

//---------------------------------------------------
// Launcher
//---------------------------------------------------

void gaussianConvolution(

    const float* d_input,

    float* d_output,

    int width,

    int height,

    int kernelSize

)
{
    dim3 block(BLOCK_SIZE,BLOCK_SIZE);

    dim3 grid(

        (width+BLOCK_SIZE-1)/BLOCK_SIZE,

        (height+BLOCK_SIZE-1)/BLOCK_SIZE

    );

    gaussianKernel<<<grid,block>>>(
        d_input,
        d_output,
        width,
        height
    );
}
