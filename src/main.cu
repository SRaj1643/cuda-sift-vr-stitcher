#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "config.h"
#include "camera_manager.h"
#include "gaussian.h"

using namespace std;
using namespace std::chrono;

int main()
{
    //------------------------------------------------------------
    // Camera Manager
    //------------------------------------------------------------

    CameraManager manager;

    manager.active_cameras = 2;

    manager.image_paths[0] = "data/img_1.jpeg";
    manager.image_paths[1] = "data/img_2.jpeg";

    //------------------------------------------------------------
    // Load Images
    //------------------------------------------------------------

    cv::Mat img1 = cv::imread(
        manager.image_paths[0],
        cv::IMREAD_GRAYSCALE
    );

    cv::Mat img2 = cv::imread(
        manager.image_paths[1],
        cv::IMREAD_GRAYSCALE
    );

    if(img1.empty() || img2.empty())
    {
        std::cout << "Error loading images." << std::endl;
        return -1;
    }

    std::cout << "Images Loaded Successfully\n";

    //------------------------------------------------------------
    // CPU Gaussian Pyramid
    //------------------------------------------------------------

    auto cpuStart = high_resolution_clock::now();

    auto cpuPyramid =
        buildGaussianPyramidCPU(img1);

    auto cpuEnd = high_resolution_clock::now();

    auto cpuTime =
        duration_cast<milliseconds>(
            cpuEnd-cpuStart
        );

    std::cout
        << "CPU Gaussian Time : "
        << cpuTime.count()
        << " ms\n";

    //------------------------------------------------------------
    // CUDA Gaussian Pyramid
    //------------------------------------------------------------

    auto gpuStart = high_resolution_clock::now();

    auto gpuPyramid =
        buildGaussianPyramidCUDA(img1);

    auto gpuEnd = high_resolution_clock::now();

    auto gpuTime =
        duration_cast<milliseconds>(
            gpuEnd-gpuStart
        );

    std::cout
        << "GPU Gaussian Time : "
        << gpuTime.count()
        << " ms\n";

    //------------------------------------------------------------
    // Save Results
    //------------------------------------------------------------

    saveGaussianPyramid(
        cpuPyramid,
        "results/cpu"
    );

    saveGaussianPyramid(
        gpuPyramid,
        "results/gpu"
    );

    std::cout
        << "Gaussian Pyramid Saved Successfully.\n";

    //------------------------------------------------------------
    // Compare CPU and GPU
    //------------------------------------------------------------

    double totalError = 0.0;
    double maxError = 0.0;

    int totalPixels = 0;

    for(int octave=0;octave<NUM_OCTAVES;octave++)
    {
        for(int scale=0;scale<NUM_SCALES;scale++)
        {
            cv::Mat diff;

            cv::absdiff(
                cpuPyramid[octave][scale],
                gpuPyramid[octave][scale],
                diff
            );

            double localMax;

            cv::minMaxLoc(
                diff,
                nullptr,
                &localMax
            );

            totalError += cv::sum(diff)[0];

            totalPixels += diff.rows*diff.cols;

            if(localMax>maxError)
                maxError=localMax;
        }
    }

    std::cout
        << "Average Absolute Error : "
        << totalError/totalPixels
        << std::endl;

    std::cout
        << "Maximum Error : "
        << maxError
        << std::endl;

    //------------------------------------------------------------
    // Speedup
    //------------------------------------------------------------

    std::cout
        << "Speedup : "
        << (double)cpuTime.count()/gpuTime.count()
        << "x\n";

    return 0;
}
