cmake_minimum_required(VERSION 3.18)

project(CUDASIFTVR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Automatically find every source file
file(GLOB_RECURSE PROJECT_SOURCES
    CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cu
    src/*.c
)

# Create executable
add_executable(cuda_sift
    ${PROJECT_SOURCES}
)

# Enable separable compilation
set_target_properties(cuda_sift PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
