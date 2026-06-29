#include "gaussian.h"

#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>

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
buildGaussianPyramidCUDA(
    const cv::Mat& image
)
{
    std::vector<std::vector<cv::Mat>> pyramid;

    cv::cuda::GpuMat currentGPU;

    image.convertTo(
        currentGPU,
        CV_32F
    );

    for(int octave=0; octave<NUM_OCTAVES; octave++)
    {
        std::vector<cv::Mat> octaveImages;

        for(int s=0; s<NUM_SCALES; s++)
        {
            cv::cuda::GpuMat blurredGPU;

            auto gaussian =
                cv::cuda::createGaussianFilter(
                    CV_32F,
                    CV_32F,
                    cv::Size(0,0),
                    SIGMAS[s]
                );

            gaussian->apply(
                currentGPU,
                blurredGPU
            );

            cv::Mat cpuImage;

            blurredGPU.download(cpuImage);

            octaveImages.push_back(cpuImage);
        }

        pyramid.push_back(octaveImages);

        cv::cuda::GpuMat nextGPU;

        cv::cuda::resize(
            currentGPU,
            nextGPU,
            cv::Size(
                currentGPU.cols/2,
                currentGPU.rows/2
            ),
            0,
            0,
            cv::INTER_AREA
        );

        currentGPU = nextGPU;
    }

    return pyramid;
}
