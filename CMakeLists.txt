cmake_minimum_required(VERSION 3.18)

project(CUDASIFTVR LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE PROJECT_SOURCES
    src/*.cpp
    src/*.cu
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
