#include "cuda_kernels.cuh"
#include <cmath>
#include <cstdio>

// =====================================================
// CONSTANT MEMORY FOR GAUSSIAN KERNEL
// =====================================================
#define MAX_KERNEL_SIZE 32
__constant__ float d_kernel[MAX_KERNEL_SIZE];
__constant__ int d_kernel_radius = 0;

// =====================================================
// HELPER: GENERATE 1D GAUSSIAN KERNEL
// =====================================================
void generateGaussianKernel(float* kernel, int radius, float sigma) {
    float sum = 0.0f;
    
    for (int i = -radius; i <= radius; i++) {
        int idx = i + radius;
        float val = expf(-((float)i * i) / (2.0f * sigma * sigma));
        kernel[idx] = val;
        sum += val;
    }
    
    // Normalize kernel
    for (int i = 0; i < 2 * radius + 1; i++) {
        kernel[i] /= sum;
    }
}

// =====================================================
// HELPER: CALCULATE KERNEL RADIUS FROM SIGMA
// =====================================================
int getKernelRadius(float sigma) {
    // Rule: kernel radius = ceil(3 * sigma)
    int radius = (int)ceil(3.0f * sigma);
    if (radius > MAX_KERNEL_SIZE / 2) {
        radius = MAX_KERNEL_SIZE / 2;
    }
    return radius;
}

// =====================================================
// HORIZONTAL GAUSSIAN BLUR KERNEL
// =====================================================
__global__ void gaussianBlurHorizontal(
    float* src,
    float* dst,
    int width,
    int height,
    float sigma
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int kernel_radius = 8;  // Will be set properly in host code
    float sum = 0.0f;
    
    // Apply horizontal convolution
    for (int i = -kernel_radius; i <= kernel_radius; i++) {
        int x = col + i;
        
        // Clamp to image boundaries (mirror padding)
        if (x < 0) x = -x;
        if (x >= width) x = 2 * (width - 1) - x;
        
        // Read from kernel constant memory
        int kernel_idx = i + kernel_radius;
        float kernel_val = d_kernel[kernel_idx];
        
        sum += src[row * width + x] * kernel_val;
    }
    
    dst[row * width + col] = sum;
}

// =====================================================
// VERTICAL GAUSSIAN BLUR KERNEL
// =====================================================
__global__ void gaussianBlurVertical(
    float* src,
    float* dst,
    int width,
    int height,
    float sigma
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int kernel_radius = 8;  // Will be set properly in host code
    float sum = 0.0f;
    
    // Apply vertical convolution
    for (int i = -kernel_radius; i <= kernel_radius; i++) {
        int y = row + i;
        
        // Clamp to image boundaries (mirror padding)
        if (y < 0) y = -y;
        if (y >= height) y = 2 * (height - 1) - y;
        
        // Read from kernel constant memory
        int kernel_idx = i + kernel_radius;
        float kernel_val = d_kernel[kernel_idx];
        
        sum += src[y * width + col] * kernel_val;
    }
    
    dst[row * width + col] = sum;
}

// =====================================================
// DOWNSAMPLING KERNEL (by factor of 2)
// =====================================================
__global__ void downsampleKernel(
    float* src,
    float* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= dst_width || row >= dst_height) return;
    
    // Sample 4 neighbors and average (simple box filter)
    int src_row = row * 2;
    int src_col = col * 2;
    
    float val = 0.0f;
    int count = 0;
    
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int y = src_row + dy;
            int x = src_col + dx;
            
            if (y < src_height && x < src_width) {
                val += src[y * src_width + x];
                count++;
            }
        }
    }
    
    if (count > 0) {
        dst[row * dst_width + col] = val / (float)count;
    }
}
