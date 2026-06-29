#include "gaussian.h"
#include "gaussian_kernel.cuh"

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

const float SIGMAS[NUM_SCALES] =
{
    1.6f,
    2.0f,
    2.52f,
    3.17f,
    4.0f,
    5.04f,
    6.35f,
    8.0f
};


//----------------------------------------------------
// SAME AS OpenCV
//----------------------------------------------------

static int computeKernelSize(float sigma)
{
    return 2 * static_cast<int>(std::ceil(3.0f * sigma)) + 1;
}


//----------------------------------------------------
// Generate Gaussian Kernel
//----------------------------------------------------

static std::vector<float>
generateGaussianKernel(float sigma)
{
    int ksize = computeKernelSize(sigma);

    int radius = ksize / 2;

    std::vector<float> kernel(
        ksize * ksize
    );

    float sum = 0.0f;

    for(int y=-radius;y<=radius;y++)
    {
        for(int x=-radius;x<=radius;x++)
        {
            float value =
                exp(-(x*x+y*y)/(2*sigma*sigma));

            kernel[
                (y+radius)*ksize+(x+radius)
            ] = value;

            sum += value;
        }
    }

    for(float &v : kernel)
        v /= sum;

    return kernel;
}


//----------------------------------------------------
// CPU Reference
//----------------------------------------------------

std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCPU(
    const cv::Mat& image
)
{
    std::vector<std::vector<cv::Mat>> pyramid;

    cv::Mat current;

    image.convertTo(current,CV_32F);

    for(int octave=0;octave<NUM_OCTAVES;octave++)
    {
        std::vector<cv::Mat> octaveImages;

        for(int s=0;s<NUM_SCALES;s++)
        {
            cv::Mat blur;

            cv::GaussianBlur(
                current,
                blur,
                cv::Size(0,0),
                SIGMAS[s]
            );

            octaveImages.push_back(blur);
        }

        pyramid.push_back(octaveImages);

        if(octave!=NUM_OCTAVES-1)
        {
            cv::resize(
                current,
                current,
                cv::Size(
                    current.cols/2,
                    current.rows/2
                ),
                0,
                0,
                cv::INTER_AREA
            );
        }
    }

    return pyramid;
}


//----------------------------------------------------
// CUDA Version
//----------------------------------------------------

std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCUDA(
    const cv::Mat& image
)
{
    std::vector<std::vector<cv::Mat>> pyramid;

    cv::Mat current;

    image.convertTo(current,CV_32F);

    for(int octave=0;octave<NUM_OCTAVES;octave++)
    {
        int width=current.cols;

        int height=current.rows;

        float* d_input;

        float* d_output;

        cudaMalloc(
            &d_input,
            width*height*sizeof(float)
        );

        cudaMalloc(
            &d_output,
            width*height*sizeof(float)
        );

        cudaMemcpy(
            d_input,
            current.ptr<float>(),
            width*height*sizeof(float),
            cudaMemcpyHostToDevice
        );

        std::vector<cv::Mat> octaveImages;

        for(int s=0;s<NUM_SCALES;s++)
        {
            auto kernel=
                generateGaussianKernel(
                    SIGMAS[s]
                );

            int kernelSize=
                computeKernelSize(
                    SIGMAS[s]
                );

            uploadGaussianKernel(
                kernel.data(),
                kernelSize
            );

            gaussianConvolution(
                d_input,
                d_output,
                width,
                height,
                kernelSize
            );

            cv::Mat blur(
                height,
                width,
                CV_32F
            );

            cudaMemcpy(
                blur.ptr<float>(),
                d_output,
                width*height*sizeof(float),
                cudaMemcpyDeviceToHost
            );

            octaveImages.push_back(blur);
        }

        pyramid.push_back(
            octaveImages
        );

        cudaFree(d_input);

        cudaFree(d_output);

        if(octave!=NUM_OCTAVES-1)
        {
            cv::resize(
                current,
                current,
                cv::Size(
                    width/2,
                    height/2
                ),
                0,
                0,
                cv::INTER_AREA
            );
        }
    }

    return pyramid;
}
