#pragma once

#include <cuda_runtime.h>

class MemoryManager
{
public:

    static float* allocateFloatArray(int size);

    static unsigned char* allocateByteArray(int size);

    static int* allocateIntArray(int size);

    static void freeMemory(void* ptr);

    static void copyToGPU(
        void* gpu_ptr,
        const void* cpu_ptr,
        size_t bytes);

    static void copyToCPU(
        void* cpu_ptr,
        const void* gpu_ptr,
        size_t bytes);

    static void deviceSynchronize();
};
