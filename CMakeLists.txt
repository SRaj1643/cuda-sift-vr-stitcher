cmake_minimum_required(VERSION 3.18)

project(CUDASIFTVR LANGUAGES CXX CUDA)

# =====================================
# Language Standards
# =====================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =====================================
# Automatically collect source files
# =====================================

file(GLOB_RECURSE PROJECT_SOURCES
    CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cu
    src/*.c
)

# =====================================
# Create executable
# =====================================

add_executable(cuda_sift
    ${PROJECT_SOURCES}
)

# =====================================
# Include directories
# =====================================

target_include_directories(cuda_sift
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# =====================================
# CUDA properties
# =====================================

set_target_properties(cuda_sift PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# =====================================
# Compiler warnings
# =====================================

target_compile_options(cuda_sift PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)
