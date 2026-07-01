#include "cuda_kernels.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

//==========================================================
// BUILD GAUSSIAN PYRAMID
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
)
{
    cudaError_t err;

    //------------------------------------------------------
    // Allocate host-side metadata
    //------------------------------------------------------

    *d_pyramid =
        new float*[num_octaves];

    *pyramid_widths =
        new int[num_octaves];

    *pyramid_heights =
        new int[num_octaves];

    //------------------------------------------------------
    // Current octave image
    //------------------------------------------------------

    float* d_current = nullptr;

    err = cudaMalloc(
        (void**)&d_current,
        input_width * input_height * sizeof(float)
    );
    
    if(err != cudaSuccess)
        return err;
    
    err = cudaMemcpy(
        d_current,
        d_input,
        input_width * input_height * sizeof(float),
        cudaMemcpyDeviceToDevice
    );
    
    if(err != cudaSuccess)
        return err;

    int current_width =
        input_width;

    int current_height =
        input_height;

    //------------------------------------------------------
    // Build each octave
    //------------------------------------------------------

    for(int octave=0;
        octave<num_octaves;
        octave++)
    {
        printf(
            "Building Octave %d (%d x %d)\n",
            octave,
            current_width,
            current_height
        );

        (*pyramid_widths)[octave] =
            current_width;

        (*pyramid_heights)[octave] =
            current_height;

        int totalPixels =
            current_width *
            current_height;

        //--------------------------------------------------
        // Allocate GPU memory for all scales
        //--------------------------------------------------

        err = cudaMalloc(
            (void**)&((*d_pyramid)[octave]),
            totalPixels *
            num_sigmas *
            sizeof(float)
        );

        if(err != cudaSuccess)
        {
            printf(
                "GPU allocation failed\n"
            );

            return err;
        }

        //--------------------------------------------------
        // Temporary buffer
        //--------------------------------------------------

        float* d_temp;

        err = cudaMalloc(
            (void**)&d_temp,
            totalPixels *
            sizeof(float)
        );

        if(err != cudaSuccess)
        {
            return err;
        }

        //--------------------------------------------------
        // Remaining implementation
        // (Part 2)
        //--------------------------------------------------
                //--------------------------------------------------
        // Build all Gaussian scales
        //--------------------------------------------------

        for(int scale = 0; scale < num_sigmas; scale++)
        {
            float sigma = sigmas[scale];

            float* d_output =
                (*d_pyramid)[octave] +
                scale * totalPixels;

            printf(
                "   Scale %d  Sigma %.2f\n",
                scale,
                sigma
            );

            err = cudaGaussianBlur(
                d_current,
                d_output,
                d_temp,
                current_width,
                current_height,
                sigma
            );

            if(err != cudaSuccess)
            {
                cudaFree(d_temp);
                cudaFree(d_current);

                return err;
            }

            err = cudaDeviceSynchronize();

            if(err != cudaSuccess)
            {
                cudaFree(d_temp);
                cudaFree(d_current);

                return err;
            }
        }

        //--------------------------------------------------
        // Free temporary buffer
        //--------------------------------------------------

        cudaFree(d_temp);

        //--------------------------------------------------
        // Remaining implementation
        // (Part 3)
        //--------------------------------------------------
                //--------------------------------------------------
        // Build next octave
        //--------------------------------------------------

        if(octave < num_octaves - 1)
        {
            int next_width  = current_width  / 2;
            int next_height = current_height / 2;

            float* d_next = nullptr;

            err = cudaMalloc(
                (void**)&d_next,
                next_width *
                next_height *
                sizeof(float)
            );

            if(err != cudaSuccess)
            {
                cudaFree(d_current);
                return err;
            }

            err = cudaDownsample(
                d_current,
                d_next,
                current_width,
                current_height,
                next_width,
                next_height
            );

            if(err != cudaSuccess)
            {
                cudaFree(d_next);
                cudaFree(d_current);
                return err;
            }

            err = cudaDeviceSynchronize();

            if(err != cudaSuccess)
            {
                cudaFree(d_next);
                cudaFree(d_current);
                return err;
            }

            //--------------------------------------------------
            // Free previous octave base image
            //--------------------------------------------------

            cudaFree(d_current);

            //--------------------------------------------------
            // Move to next octave
            //--------------------------------------------------

            d_current = d_next;

            current_width  = next_width;
            current_height = next_height;
        }
    }
            //------------------------------------------------------
    // Cleanup Final Octave Base Image
    //------------------------------------------------------

    if(d_current != nullptr)
    {
        cudaFree(d_current);
        d_current = nullptr;
    }

    //------------------------------------------------------
    // Gaussian Pyramid Built Successfully
    //------------------------------------------------------

    printf("\n");
    printf("=====================================\n");
    printf(" Gaussian Pyramid Complete\n");
    printf("=====================================\n");

    return cudaSuccess;
}

//==========================================================
// FREE GAUSSIAN PYRAMID
//==========================================================

void freePyramid(
    float** d_pyramid,
    int num_octaves,
    int* pyramid_widths,
    int* pyramid_heights
)
{
    if(d_pyramid != nullptr)
    {
        for(int octave=0;
            octave<num_octaves;
            octave++)
        {
            if(d_pyramid[octave]!=nullptr)
            {
                cudaFree(
                    d_pyramid[octave]
                );
            }
        }

        delete[] d_pyramid;
    }

    if(pyramid_widths!=nullptr)
        delete[] pyramid_widths;

    if(pyramid_heights!=nullptr)
        delete[] pyramid_heights;
}
