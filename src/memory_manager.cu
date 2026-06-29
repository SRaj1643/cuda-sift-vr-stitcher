#include "memory_manager.h"

float* MemoryManager::allocateFloatArray(int size)
{
    float* ptr = nullptr;

    cudaMalloc(&ptr,
               size * sizeof(float));

    return ptr;
}

unsigned char*
MemoryManager::allocateByteArray(int size)
{
    unsigned char* ptr = nullptr;

    cudaMalloc(&ptr,
               size * sizeof(unsigned char));

    return ptr;
}

int*
MemoryManager::allocateIntArray(int size)
{
    int* ptr = nullptr;

    cudaMalloc(&ptr,
               size * sizeof(int));

    return ptr;
}

void MemoryManager::freeMemory(void* ptr)
{
    if(ptr)
        cudaFree(ptr);
}

void MemoryManager::copyToGPU(
    void* gpu_ptr,
    const void* cpu_ptr,
    size_t bytes)
{
    cudaMemcpy(
        gpu_ptr,
        cpu_ptr,
        bytes,
        cudaMemcpyHostToDevice);
}

void MemoryManager::copyToCPU(
    void* cpu_ptr,
    const void* gpu_ptr,
    size_t bytes)
{
    cudaMemcpy(
        cpu_ptr,
        gpu_ptr,
        bytes,
        cudaMemcpyDeviceToHost);
}

void MemoryManager::deviceSynchronize()
{
    cudaDeviceSynchronize();
}
