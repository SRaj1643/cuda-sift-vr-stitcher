#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

// =====================================================
// GAUSSIAN BLUR KERNELS (Separable)
// =====================================================

// Horizontal Gaussian blur kernel
// Input: src device image, Output: dst device image
// width, height: image dimensions
// sigma: standard deviation for Gaussian kernel
__global__ void gaussianBlurHorizontal(
    float* src,
    float* dst,
    int width,
    int height,
    float sigma
);

// Vertical Gaussian blur kernel
// Input: src device image, Output: dst device image
__global__ void gaussianBlurVertical(
    float* src,
    float* dst,
    int width,
    int height,
    float sigma
);

// =====================================================
// DOWNSAMPLING KERNEL
// =====================================================

// Downsample image by factor of 2
// Input: src (original size), Output: dst (half size)
// src_width, src_height: original dimensions
// dst_width, dst_height: half dimensions
__global__ void downsampleKernel(
    float* src,
    float* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
);

// =====================================================
// HELPER FUNCTIONS
// =====================================================

// Generate 1D Gaussian kernel coefficients
void generateGaussianKernel(
    float* kernel,
    int radius,
    float sigma
);

// Calculate radius based on sigma
int getKernelRadius(float sigma);

#endif // CUDA_KERNELS_CUH
