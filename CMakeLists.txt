cmake_minimum_required(VERSION 3.18)

project(CUDASIFTVR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

enable_language(CUDA)

add_executable(

    cuda_sift

    src/main.cu

    src/memory_manager.cu

    src/test_kernel.cu
)
