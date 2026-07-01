#ifndef IMAGE_UTILS_CUH
#define IMAGE_UTILS_CUH

#include <cuda_runtime.h>

// =====================================================
// IMAGE UTILITIES
// =====================================================

// Load image from file (PPM format for simplicity)
// Returns pointer to float array on CPU
float* loadImagePPM(const char* filename, int& width, int& height);

// Save image to file (PPM format)
bool saveImagePPM(const char* filename, float* data, int width, int height);

// Convert uint8 image to float32 (0-255 -> 0-1)
void convertToFloat32(
    unsigned char* src,
    float* dst,
    int size
);

// Convert float32 image to uint8 (0-1 -> 0-255)
void convertToUint8(
    float* src,
    unsigned char* dst,
    int size
);

// Allocate GPU memory and copy image
cudaError_t uploadImageGPU(
    float* h_image,
    float** d_image,
    int width,
    int height
);

// Download image from GPU memory
cudaError_t downloadImageGPU(
    float* d_image,
    float* h_image,
    int width,
    int height
);

#endif // IMAGE_UTILS_CUH
