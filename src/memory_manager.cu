#pragma once

#include <cuda_runtime.h>

class MemoryManager
{
public:

    static float* allocateFloatArray(int size);

    static void freeMemory(float* ptr);

    static void copyToGPU(
        float* gpu_ptr,
        float* cpu_ptr,
        int size);

    static void copyToCPU(
        float* cpu_ptr,
        float* gpu_ptr,
        int size);
};
