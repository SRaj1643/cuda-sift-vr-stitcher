#include "cuda_kernels.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>

// =====================================================
// GAUSSIAN BLUR FUNCTION (wrapper)
// =====================================================
cudaError_t cudaGaussianBlur(
    float* d_input,
    float* d_output,
    float* d_temp,
    int width,
    int height,
    float sigma
) {
    // Step 1: Generate Gaussian kernel on CPU
    int radius = (int)ceil(3.0f * sigma);
    if (radius > 15) radius = 15;  // Limit to fit in constant memory
    
    float* h_kernel = new float[2 * radius + 1];
    generateGaussianKernel(h_kernel, radius, sigma);
    
    // Step 2: Copy kernel to constant memory
    cudaError_t err = cudaMemcpyToSymbol(d_kernel, h_kernel, (2 * radius + 1) * sizeof(float));
    if (err != cudaSuccess) {
        printf("Error copying kernel to constant memory: %s\n", cudaGetErrorString(err));
        delete[] h_kernel;
        return err;
    }
    
    // Step 3: Setup grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );
    
    // Step 4: Horizontal blur
    gaussianBlurHorizontal<<<gridDim, blockDim>>>(d_input, d_temp, width, height, sigma);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Horizontal blur kernel error: %s\n", cudaGetErrorString(err));
        delete[] h_kernel;
        return err;
    }
    
    // Step 5: Vertical blur
    gaussianBlurVertical<<<gridDim, blockDim>>>(d_temp, d_output, width, height, sigma);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Vertical blur kernel error: %s\n", cudaGetErrorString(err));
        delete[] h_kernel;
        return err;
    }
    
    // Cleanup
    delete[] h_kernel;
    return cudaSuccess;
}

// =====================================================
// DOWNSAMPLE FUNCTION (wrapper)
// =====================================================
cudaError_t cudaDownsample(
    float* d_input,
    float* d_output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height
) {
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (dst_width + blockDim.x - 1) / blockDim.x,
        (dst_height + blockDim.y - 1) / blockDim.y
    );
    
    downsampleKernel<<<gridDim, blockDim>>>(
        d_input, d_output,
        src_width, src_height,
        dst_width, dst_height
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Downsample kernel error: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    return cudaSuccess;
}

// =====================================================
// BUILD GAUSSIAN PYRAMID
// =====================================================
cudaError_t buildGaussianPyramid(
    float* d_input,
    int input_width,
    int input_height,
    int num_octaves,
    float* sigmas,
    int num_sigmas,
    float*** d_pyramid,  // Output: pyramid[octave][scale]
    int** pyramid_widths,
    int** pyramid_heights
) {
    cudaError_t err;
    
    // Allocate pyramid structure on CPU
    *d_pyramid = new float*[num_octaves];
    *pyramid_widths = new int[num_octaves];
    *pyramid_heights = new int[num_octaves];
    
    float* d_current = d_input;
    int current_width = input_width;
    int current_height = input_height;
    
    // Build each octave
    for (int octave = 0; octave < num_octaves; octave++) {
        printf("Building octave %d (%d x %d)...\n", octave, current_width, current_height);
        
        // Allocate array for this octave's scales
        (*d_pyramid)[octave] = new float[num_sigmas * current_width * current_height];
        (*pyramid_widths)[octave] = current_width;
        (*pyramid_heights)[octave] = current_height;
        
        int total_size = current_width * current_height;
        
        // Allocate temporary buffer for separable blur
        float* d_temp;
        err = cudaMalloc(&d_temp, total_size * sizeof(float));
        if (err != cudaSuccess) {
            printf("Error allocating temp buffer: %s\n", cudaGetErrorString(err));
            return err;
        }
        
        // Apply each sigma
        for (int s = 0; s < num_sigmas; s++) {
            float sigma = sigmas[s];
            float* d_out = (*d_pyramid)[octave] + s * total_size;
            
            printf("  Sigma %.2f...\n", sigma);
            
            // Blur current image with sigma
            err = cudaGaussianBlur(d_current, d_out, d_temp, current_width, current_height, sigma);
            if (err != cudaSuccess) {
                cudaFree(d_temp);
                return err;
            }
        }
        
        // Free temp buffer
        cudaFree(d_temp);
        
        // Downsample for next octave
        if (octave < num_octaves - 1) {
            int next_width = current_width / 2;
            int next_height = current_height / 2;
            
            float* d_downsampled;
            int downsample_size = next_width * next_height;
            
            err = cudaMalloc(&d_downsampled, downsample_size * sizeof(float));
            if (err != cudaSuccess) {
                printf("Error allocating downsample buffer: %s\n", cudaGetErrorString(err));
                return err;
            }
            
            // Downsample the first scale of current octave
            err = cudaDownsample(
                (*d_pyramid)[octave],  // Use first blurred image
                d_downsampled,
                current_width, current_height,
                next_width, next_height
            );
            if (err != cudaSuccess) {
                cudaFree(d_downsampled);
                return err;
            }
            
            // Free previous octave data if not the input
            if (octave > 0) {
                cudaFree(d_current);
            }
            
            d_current = d_downsampled;
            current_width = next_width;
            current_height = next_height;
        }
    }
    
    // Free final downsampled image
    if (num_octaves > 1) {
        cudaFree(d_current);
    }
    
    return cudaSuccess;
}

// =====================================================
// FREE PYRAMID MEMORY
// =====================================================
void freePyramid(
    float** d_pyramid,
    int num_octaves,
    int* pyramid_widths,
    int* pyramid_heights
) {
    for (int i = 0; i < num_octaves; i++) {
        cudaFree(d_pyramid[i]);
    }
    delete[] d_pyramid;
    delete[] pyramid_widths;
    delete[] pyramid_heights;
}
