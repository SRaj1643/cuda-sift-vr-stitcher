#include <cuda_runtime.h>

__global__
void multiplyByTwoKernel(float* data, int size)
{
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if(idx < size)
    {
        data[idx] *= 2.0f;
    }
}
