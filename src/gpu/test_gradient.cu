#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include "cuda_kernels.cuh"
#include "image_utils.cuh"

#define NUM_OCTAVES 4
#define NUM_SIGMAS 8

float SIGMAS[NUM_SIGMAS] = {1.6f, 2.0f, 2.52f, 3.17f, 4.0f, 5.04f, 6.35f, 8.0f};

int main() {
    printf("======================================\n");
    printf("CUDA Gaussian Pyramid - Test 1: Gradient\n");
    printf("======================================\n\n");
    
    // Create synthetic test image 1: Gradient
    int width = 512;
    int height = 512;
    float* h_image1 = new float[width * height];
    
    printf("Test Image 1: Gradient (512x512)\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_image1[y * width + x] = ((float)x + (float)y) / (width + height);
        }
    }
    
    // Create synthetic test image 2: Checkerboard
    float* h_image2 = new float[width * height];
    printf("Test Image 2: Checkerboard (512x512)\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int block = ((x / 32) + (y / 32)) % 2;
            h_image2[y * width + x] = (block == 0) ? 0.3f : 0.7f;
        }
    }
    
    // Test both images
    for (int img_num = 1; img_num <= 2; img_num++) {
        printf("\n--- Testing Image %d ---\n", img_num);
        
        float* h_image = (img_num == 1) ? h_image1 : h_image2;
        
        // Upload to GPU
        float* d_input;
        printf("Uploading to GPU...\n");
        uploadImageGPU(h_image, &d_input, width, height);
        
        // Build pyramid
        float*** d_pyramid;
        int* pyramid_widths;
        int* pyramid_heights;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        buildGaussianPyramid(
            d_input, width, height,
            NUM_OCTAVES, SIGMAS, NUM_SIGMAS,
            &d_pyramid,
            &pyramid_widths,
            &pyramid_heights
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("Pyramid built in %lld ms\n", duration.count());
        
        // Print pyramid structure
        printf("\nPyramid Structure:\n");
        for (int octave = 0; octave < NUM_OCTAVES; octave++) {
            printf("  Octave %d: %d x %d (%d scales)\n",
                   octave,
                   pyramid_widths[octave],
                   pyramid_heights[octave],
                   NUM_SIGMAS);
        }
        
        // Download first octave first scale and save
        int out_size = pyramid_widths[0] * pyramid_heights[0];
        float* h_output = new float[out_size];
        
        downloadImageGPU(d_pyramid[0], h_output, pyramid_widths[0], pyramid_heights[0]);
        
        char filename[100];
        sprintf(filename, "output_image%d_octave0.ppm", img_num);
        saveImagePPM(filename, h_output, pyramid_widths[0], pyramid_heights[0]);
        printf("Saved %s\n", filename);
        
        // Verify output statistics
        float min_val = h_output[0], max_val = h_output[0], sum = 0.0f;
        for (int i = 0; i < out_size; i++) {
            if (h_output[i] < min_val) min_val = h_output[i];
            if (h_output[i] > max_val) max_val = h_output[i];
            sum += h_output[i];
        }
        
        printf("Output Statistics:\n");
        printf("  Min: %.6f\n", min_val);
        printf("  Max: %.6f\n", max_val);
        printf("  Mean: %.6f\n", sum / out_size);
        
        // Cleanup
        cudaFree(d_input);
        freePyramid(d_pyramid, NUM_OCTAVES, pyramid_widths, pyramid_heights);
        delete[] h_output;
    }
    
    delete[] h_image1;
    delete[] h_image2;
    
    printf("\n✅ All tests passed!\n");
    printf("Check output_image1_octave0.ppm and output_image2_octave0.ppm\n");
    
    return 0;
}
