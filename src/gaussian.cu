#include "gaussian.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

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


//==========================================================
// CPU VERSION
//==========================================================

std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCPU(
    const cv::Mat& image)
{
    std::vector<std::vector<cv::Mat>> pyramid;

    cv::Mat current;

    image.convertTo(current,CV_32F);

    for(int octave=0;octave<NUM_OCTAVES;octave++)
    {
        std::vector<cv::Mat> octaveImages;

        for(int scale=0;scale<NUM_SCALES;scale++)
        {
            cv::Mat blurred;

            cv::GaussianBlur(
                current,
                blurred,
                cv::Size(0,0),
                SIGMAS[scale]
            );

            octaveImages.push_back(blurred);
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


//==========================================================
// CUDA VERSION
//==========================================================

std::vector<std::vector<cv::Mat>>
buildGaussianPyramidCUDA(
    const cv::Mat& image)
{
    std::vector<std::vector<cv::Mat>> pyramid;

    cv::Mat floatImage;

    image.convertTo(
        floatImage,
        CV_32F
    );

    cv::cuda::GpuMat currentGPU;

    currentGPU.upload(floatImage);

    std::vector<cv::Ptr<cv::cuda::Filter>> gaussianFilters;

    for(int i=0;i<NUM_SCALES;i++)
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

    for(int octave=0;octave<NUM_OCTAVES;octave++)
    {
        std::vector<cv::Mat> octaveImages;

        for(int scale=0;scale<NUM_SCALES;scale++)
        {
            cv::cuda::GpuMat blurredGPU;

            gaussianFilters[scale]->apply(
                currentGPU,
                blurredGPU
            );

            cv::Mat blurredCPU;

            blurredGPU.download(
                blurredCPU
            );

            octaveImages.push_back(
                blurredCPU
            );
        }

        pyramid.push_back(
            octaveImages
        );

        if(octave!=NUM_OCTAVES-1)
        {
            cv::cuda::GpuMat resizedGPU;

            cv::cuda::resize(
                currentGPU,
                resizedGPU,
                cv::Size(
                    currentGPU.cols/2,
                    currentGPU.rows/2
                ),
                0,
                0,
                cv::INTER_AREA
            );

            currentGPU = resizedGPU;
        }
    }

    return pyramid;
}


//==========================================================
// SAVE PYRAMID
//==========================================================

void saveGaussianPyramid(
    const std::vector<std::vector<cv::Mat>>& pyramid,
    const std::string& folder)
{
    for(size_t octave=0;octave<pyramid.size();octave++)
    {
        for(size_t scale=0;scale<pyramid[octave].size();scale++)
        {
            cv::Mat image8U;

            cv::normalize(

                pyramid[octave][scale],

                image8U,

                0,

                255,

                cv::NORM_MINMAX

            );

            image8U.convertTo(
                image8U,
                CV_8U
            );

            std::string filename=

                folder+

                "/octave"+

                std::to_string(octave)+

                "_scale"+

                std::to_string(scale)+

                ".png";

            cv::imwrite(
                filename,
                image8U
            );
        }
    }
}
