#include <iostream>

#include "memory_manager.h"

void runMemoryTest()
{
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

    float cpu_result[SIZE];

    MemoryManager::copyToCPU(
        cpu_result,
        gpu_data,
        SIZE);

    std::cout << "\nMemory Test\n";

    for(int i = 0; i < SIZE; i++)
    {
        std::cout << cpu_result[i] << " ";
    }

    std::cout << std::endl;

    MemoryManager::freeMemory(gpu_data);
}
