#pragma once

#include <cuda_runtime.h>

class MemoryManager
{
public:

    static float*
    allocateFloatArray(size_t size);

    static void
    freeMemory(void* ptr);

    static void
    copyToGPU(
        void* gpu,
        const void* cpu,
        size_t bytes);

    static void
    copyToCPU(
        void* cpu,
        const void* gpu,
        size_t bytes);

    static void
    synchronize();

};
