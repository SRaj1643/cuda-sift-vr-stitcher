#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include "cuda_kernels.cuh"
#include "image_utils.cuh"
#include "gaussian_pyramid.cuh"
// Global parameters (from CPU version)
#define NUM_OCTAVES 4
#define NUM_SIGMAS 8

float SIGMAS[NUM_SIGMAS] = {
    1.6f, 2.0f, 2.52f, 3.17f,
    4.0f, 5.04f, 6.35f, 8.0f
};

// =====================================================
// TIMING UTILITY
// =====================================================
class GPUTimer {
public:
    cudaEvent_t start, stop;
    
    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void tick() {
        cudaEventRecord(start);
    }
    
    float tock() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// =====================================================
// MAIN PROGRAM
// =====================================================
int main(int argc, char** argv) {
    printf("CUDA Gaussian Pyramid Builder\n");
    printf("==============================\n\n");
    
    // Create synthetic test image (256x256 grayscale)
    int width = 256;
    int height = 256;
    float* h_image = new float[width * height];
    
    // Fill with gradient pattern for testing
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_image[y * width + x] = ((float)x + (float)y) / (width + height);
        }
    }
    
    printf("Input image: %d x %d\n", width, height);
    printf("Number of octaves: %d\n", NUM_OCTAVES);
    printf("Number of scales per octave: %d\n", NUM_SIGMAS);
    
    // =====================================================
    // UPLOAD IMAGE TO GPU
    // =====================================================
    float* d_input;
    printf("\nUploading image to GPU...\n");
    
    cudaError_t err = uploadImageGPU(h_image, &d_input, width, height);
    if (err != cudaSuccess) {
        printf("Failed to upload image\n");
        return 1;
    }
    
    // =====================================================
    // BUILD PYRAMID ON GPU
    // =====================================================
    printf("Building Gaussian pyramid on GPU...\n");
    
    float*** d_pyramid;
    int* pyramid_widths;
    int* pyramid_heights;
    
    GPUTimer timer;
    timer.tick();
    
    err = buildGaussianPyramid(
        d_input,
        width, height,
        NUM_OCTAVES,
        SIGMAS, NUM_SIGMAS,
        &d_pyramid,
        &pyramid_widths,
        &pyramid_heights
    );
    
    float build_time = timer.tock();
    
    if (err != cudaSuccess) {
        printf("Failed to build pyramid: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return 1;
    }
    
    printf("Pyramid built in %.2f ms\n\n", build_time);
    
    // =====================================================
    // VERIFY PYRAMID STRUCTURE
    // =====================================================
    printf("Pyramid structure:\n");
    for (int octave = 0; octave < NUM_OCTAVES; octave++) {
        printf("  Octave %d: %d x %d (%d scales)\n",
               octave,
               pyramid_widths[octave],
               pyramid_heights[octave],
               NUM_SIGMAS);
    }
    
    // =====================================================
    // DOWNLOAD AND SAVE FIRST OCTAVE, FIRST SCALE
    // =====================================================
    printf("\nDownloading first scale of first octave...\n");
    
    int out_width = pyramid_widths[0];
    int out_height = pyramid_heights[0];
    float* h_output = new float[out_width * out_height];
    
    err = downloadImageGPU(d_pyramid[0], h_output, out_width, out_height);
    if (err != cudaSuccess) {
        printf("Failed to download image: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Save output
    if (saveImagePPM("output_octave0_scale0.ppm", h_output, out_width, out_height)) {
        printf("Saved output_octave0_scale0.ppm\n");
    }
    
    // =====================================================
    // CLEANUP
    // =====================================================
    printf("\nCleaning up...\n");
    
    cudaFree(d_input);
    freePyramid(d_pyramid, NUM_OCTAVES, pyramid_widths, pyramid_heights);
    
    delete[] h_image;
    delete[] h_output;
    
    printf("Done!\n");
    
    return 0;
}
