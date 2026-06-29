#include "gaussian_kernel.cuh"

#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define MAX_KERNEL_SIZE 31
#define MAX_RADIUS 15

//-----------------------------------------------------
// Constant Memory
//-----------------------------------------------------

__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
__constant__ int d_kernelSize;

//-----------------------------------------------------
// Upload Gaussian Kernel
//-----------------------------------------------------

void uploadGaussianKernel(
    const float* kernel,
    int kernelSize)
{
    cudaMemcpyToSymbol(
        d_kernel,
        kernel,
        kernelSize * kernelSize * sizeof(float));

    cudaMemcpyToSymbol(
        d_kernelSize,
        &kernelSize,
        sizeof(int));
}

//-----------------------------------------------------
// Shared Memory Gaussian
//-----------------------------------------------------

__global__
void gaussianKernelShared(

    const float* input,

    float* output,

    int width,

    int height)

{

    const int radius = d_kernelSize / 2;

    const int TILE = BLOCK_SIZE + 2 * MAX_RADIUS;

    __shared__ float tile[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    //-------------------------------------------------
    // Load Center Pixel
    //-------------------------------------------------

    int lx = tx + radius;
    int ly = ty + radius;

    if(x < width && y < height)
        tile[ly][lx] = input[y * width + x];
    else
        tile[ly][lx] = 0.0f;

    //-------------------------------------------------
    // Halo Loading
    //-------------------------------------------------

    if(tx < radius)
    {
        int xx = max(x - radius,0);

        tile[ly][tx] =
            input[y * width + xx];
    }

    if(tx >= BLOCK_SIZE-radius)
    {
        int xx =
            min(
                x+radius,
                width-1);

        tile[ly][tx+2*radius] =
            input[y*width+xx];
    }

    if(ty < radius)
    {
        int yy=max(y-radius,0);

        tile[ty][lx] =
            input[yy*width+x];
    }

    if(ty>=BLOCK_SIZE-radius)
    {
        int yy=
            min(
                y+radius,
                height-1);

        tile[ty+2*radius][lx]=
            input[yy*width+x];
    }

    __syncthreads();

    //-------------------------------------------------
    // Convolution
    //-------------------------------------------------

    if(x>=width||y>=height)
        return;

    float sum=0.0f;

    for(int ky=-radius;ky<=radius;ky++)
    {
        for(int kx=-radius;kx<=radius;kx++)
        {
            float pixel=
                tile[
                    ly+ky
                ][
                    lx+kx
                ];

            float weight=
                d_kernel[
                    (ky+radius)*d_kernelSize
                    +(kx+radius)
                ];

            sum+=pixel*weight;
        }
    }

    output[
        y*width+x
    ]=sum;

}
