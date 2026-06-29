#include "memory_manager.h"

float*
MemoryManager::allocateFloatArray(
    size_t size)
{
    float* ptr=nullptr;

    cudaMalloc(
        &ptr,
        size*sizeof(float));

    return ptr;
}

void
MemoryManager::freeMemory(
    void* ptr)
{
    if(ptr)
        cudaFree(ptr);
}

void
MemoryManager::copyToGPU(
    void* gpu,
    const void* cpu,
    size_t bytes)
{
    cudaMemcpy(
        gpu,
        cpu,
        bytes,
        cudaMemcpyHostToDevice);
}

void
MemoryManager::copyToCPU(
    void* cpu,
    const void* gpu,
    size_t bytes)
{
    cudaMemcpy(
        cpu,
        gpu,
        bytes,
        cudaMemcpyDeviceToHost);
}

void
MemoryManager::synchronize()
{
    cudaDeviceSynchronize();
}
