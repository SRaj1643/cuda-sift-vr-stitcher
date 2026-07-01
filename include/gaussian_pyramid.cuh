#ifndef GAUSSIAN_PYRAMID_CUH
#define GAUSSIAN_PYRAMID_CUH

#include <cuda_runtime.h>

//==========================================================
// BUILD GAUSSIAN PYRAMID
//==========================================================
//
// Input:
//   d_input          -> Input image on GPU
//   input_width      -> Image width
//   input_height     -> Image height
//   num_octaves      -> Number of octaves
//   sigmas           -> Sigma values
//   num_sigmas       -> Number of scales
//
// Output:
//   d_pyramid        -> GPU Gaussian pyramid
//   pyramid_widths   -> Width of each octave
//   pyramid_heights  -> Height of each octave
//
//==========================================================

cudaError_t buildGaussianPyramid(
    float* d_input,
    int input_width,
    int input_height,
    int num_octaves,
    float* sigmas,
    int num_sigmas,
    float*** d_pyramid,
    int** pyramid_widths,
    int** pyramid_heights
);

//==========================================================
// FREE GAUSSIAN PYRAMID
//==========================================================

void freePyramid(
    float** d_pyramid,
    int num_octaves,
    int* pyramid_widths,
    int* pyramid_heights
);

#endif
