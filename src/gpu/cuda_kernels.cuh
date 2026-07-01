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
// GAUSSIAN HELPER FUNCTIONS
//==========================================================

// Compute kernel radius from sigma
int getKernelRadius(float sigma);

// Generate normalized 1D Gaussian kernel
void generateGaussianKernel(
    float* kernel,
    int radius,
    float sigma
);

// Upload kernel to constant memory
void uploadGaussianKernel(
    float sigma
);

//==========================================================
// CUDA KERNELS
//==========================================================

// Horizontal Gaussian Pass
__global__
void gaussianBlurHorizontal(
    const float* src,
    float* dst,
    int width,
    int height
);

// Vertical Gaussian Pass
__global__
void gaussianBlurVertical(
    const float* src,
    float* dst,
    int width,
    int height
);

// Downsample by factor of 2
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
// CUDA WRAPPERS
//==========================================================

// Complete Gaussian Blur
cudaError_t cudaGaussianBlur(
    float* d_input,
    float* d_output,
    float* d_temp,
    int width,
    int height,
    float sigma
);

// Downsample Wrapper
cudaError_t cudaDownsample(
    float* d_input,
    float* d_output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
);

#endif
