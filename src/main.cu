#include <iostream>
#include "config.h"
#include "camera_manager.h"
#include "memory_manager.h"
#include "test_kernel.h"
#include "gaussian.h"

int main()
{
    CameraManager manager;

    manager.active_cameras = 2;

    const int SIZE = 10;

    float cpu_data[SIZE];

    for(int i = 0; i < SIZE; i++)
    {
        cpu_data[i] = i;
    }

    float* gpu_data =
        MemoryManager::allocateFloatArray(SIZE);

    MemoryManager::copyToGPU(
        gpu_data,
        cpu_data,
        SIZE);

    launchMultiplyByTwo(
    gpu_data,
    SIZE);

    float cpu_result[SIZE];

    MemoryManager::copyToCPU(
        cpu_result,
        gpu_data,
        SIZE);

    std::cout << "Transferred Values:\n";

    for(int i = 0; i < SIZE; i++)
    {
        std::cout << cpu_result[i] << " ";
    }

    std::cout << std::endl;

    MemoryManager::freeMemory(gpu_data);
    std::vector<float*> gaussian_pyramid;

    buildGaussianPyramid(
        nullptr,
        0,
        0,
        4,
        8,
        gaussian_pyramid
    );

    return 0;
}
