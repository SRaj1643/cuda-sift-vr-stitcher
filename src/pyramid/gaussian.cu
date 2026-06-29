#include "gaussian.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>

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

std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCUDA(const cv::Mat& image)
{
    std::vector<std::vector<cv::Mat>> pyramid;

    // Convert CPU image to float
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F);

    // Upload once
    cv::cuda::GpuMat currentGPU;
    currentGPU.upload(floatImage);

    // Create Gaussian filters ONCE
    std::vector<cv::Ptr<cv::cuda::Filter>> gaussianFilters;

    for(int i = 0; i < NUM_SCALES; i++)
    {
        gaussianFilters.push_back(
            cv::cuda::createGaussianFilter(
                CV_32F,
                CV_32F,
                cv::Size(0,0),
                SIGMAS[i]
            )
        );
    }

    for(int octave = 0; octave < NUM_OCTAVES; octave++)
    {
        std::vector<cv::Mat> octaveImages;

        for(int scale = 0; scale < NUM_SCALES; scale++)
        {
            cv::cuda::GpuMat blurredGPU;

            gaussianFilters[scale]->apply(
                currentGPU,
                blurredGPU
            );

            cv::Mat cpuImage;

            blurredGPU.download(cpuImage);

            octaveImages.push_back(cpuImage);
        }

        pyramid.push_back(octaveImages);

        if(octave != NUM_OCTAVES - 1)
        {
            cv::cuda::GpuMat nextGPU;

            cv::cuda::resize(
                currentGPU,
                nextGPU,
                cv::Size(
                    currentGPU.cols / 2,
                    currentGPU.rows / 2
                ),
                0,
                0,
                cv::INTER_AREA
            );

            currentGPU = nextGPU;
        }
    }

    return pyramid;
}
