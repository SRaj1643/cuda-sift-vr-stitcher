#include "../include/memory_manager.h"

float* MemoryManager::allocateFloatArray(int size)
{
    float* ptr = nullptr;

    cudaMalloc(
        (void**)&ptr,
        size * sizeof(float));

    return ptr;
}

void MemoryManager::freeMemory(float* ptr)
{
    if(ptr != nullptr)
    {
        cudaFree(ptr);
    }
}

void MemoryManager::copyToGPU(
    float* gpu_ptr,
    float* cpu_ptr,
    int size)
{
    cudaMemcpy(
        gpu_ptr,
        cpu_ptr,
        size * sizeof(float),
        cudaMemcpyHostToDevice);
}

void MemoryManager::copyToCPU(
    float* cpu_ptr,
    float* gpu_ptr,
    int size)
{
    cudaMemcpy(
        cpu_ptr,
        gpu_ptr,
        size * sizeof(float),
        cudaMemcpyDeviceToHost);
}
