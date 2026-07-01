#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

//==========================================================
// CONSTANTS
//==========================================================

#define MAX_KERNEL_RADIUS 24
#define MAX_KERNEL_SIZE (2 * MAX_KERNEL_RADIUS + 1)

//==========================================================
// CONSTANT MEMORY
//==========================================================

extern __constant__ float d_kernel[MAX_KERNEL_SIZE];
extern __constant__ int d_kernel_radius;

//==========================================================
// CUDA KERNELS
//==========================================================

__global__
void gaussianBlurHorizontal(
    const float* src,
    float* dst,
    int width,
    int height
);

__global__
void gaussianBlurVertical(
    const float* src,
    float* dst,
    int width,
    int height
);

__global__
void downsampleKernel(
    const float* src,
    float* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
);

//==========================================================
// HOST FUNCTIONS
//==========================================================

void generateGaussianKernel(
    float* kernel,
    int radius,
    float sigma
);

int getKernelRadius(
    float sigma
);

#endif
