cmake_minimum_required(VERSION 3.18)
project(CUDA_GaussianPyramid LANGUAGES CUDA CXX)

# =====================================================
# COMPILER SETTINGS
# =====================================================
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Architecture (adjust based on your GPU)
# sm_70: Tesla V100
# sm_75: RTX 2070/2080
# sm_86: RTX 3080/A100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

# Optimizations
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# =====================================================
# SOURCE FILES
# =====================================================
set(SOURCES
    main.cu
    cuda_kernels.cu
    gaussian_pyramid.cu
    image_utils.cu
)

set(HEADERS
    cuda_kernels.cuh
    image_utils.cuh
)

# =====================================================
# EXECUTABLE
# =====================================================
add_executable(gaussian_pyramid ${SOURCES} ${HEADERS})

# =====================================================
# INCLUDE DIRECTORIES
# =====================================================
target_include_directories(gaussian_pyramid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# =====================================================
# CUDA RUNTIME
# =====================================================
target_link_libraries(gaussian_pyramid PRIVATE cudart)

# =====================================================
# BUILD SETTINGS
# =====================================================
set_target_properties(gaussian_pyramid PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION OFF
)
