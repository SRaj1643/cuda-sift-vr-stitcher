cmake_minimum_required(VERSION 3.18)

project(CUDASIFTVR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)

# --------------------------------------------------
# Explicit Source Files
# --------------------------------------------------

set(PROJECT_SOURCES

    src/main.cu

    src/gpu/cuda_kernels.cu
    src/gpu/gaussian_pyramid.cu
    src/gpu/image_utils.cu

)

add_executable(cuda_sift
    ${PROJECT_SOURCES}
)

target_include_directories(cuda_sift PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cuda_sift
    ${OpenCV_LIBS}
)

set_target_properties(cuda_sift PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
