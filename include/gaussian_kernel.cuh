#ifndef GAUSSIAN_KERNEL_CUH
#define GAUSSIAN_KERNEL_CUH

#include <cuda_runtime.h>

#define MAX_KERNEL_SIZE 31

// Maximum supported Gaussian radius = 15
// Maximum kernel size = 31 x 31

void uploadGaussianKernel(
    const float* kernel,
    int kernelSize
);

void gaussianConvolution(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int kernelSize
);

#endif
