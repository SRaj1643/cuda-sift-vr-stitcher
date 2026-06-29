#include <iostream>

#include "config.h"
#include "camera_manager.h"
#include "memory_manager.h"
#include "gaussian.h"

int main()
{
    // --------------------------------------------------
    // Camera Manager (Future 6-Camera Architecture)
    // --------------------------------------------------

    CameraManager manager;
    manager.active_cameras = 2;

    manager.image_paths[0] = "data/img_1.jpeg";
    manager.image_paths[1] = "data/img_2.jpeg";

    // --------------------------------------------------
    // Create a Small Test Image (8x8)
    // --------------------------------------------------

    const int WIDTH = 8;
    const int HEIGHT = 8;
    const int SIZE = WIDTH * HEIGHT;

    float input_image[SIZE];

    for (int i = 0; i < SIZE; i++)
    {
        input_image[i] = static_cast<float>(i);
    }

    float output_image[SIZE];

    // --------------------------------------------------
    // Allocate GPU Memory
    // --------------------------------------------------

    float* gpu_input =
        MemoryManager::allocateFloatArray(SIZE);

    float* gpu_output =
        MemoryManager::allocateFloatArray(SIZE);

    // --------------------------------------------------
    // Copy Image to GPU
    // --------------------------------------------------

    MemoryManager::copyToGPU(
        gpu_input,
        input_image,
        SIZE
    );

    // --------------------------------------------------
    // Launch CUDA Kernel
    // (Currently Copy Kernel)
    // --------------------------------------------------

    gaussianBlurCUDA(
        gpu_input,
        gpu_output,
        WIDTH,
        HEIGHT,
        1.6f
    );

    // --------------------------------------------------
    // Copy Result Back
    // --------------------------------------------------

    MemoryManager::copyToCPU(
        output_image,
        gpu_output,
        SIZE
    );

    // --------------------------------------------------
    // Display Input Image
    // --------------------------------------------------

    std::cout << "\nInput Image\n\n";

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            std::cout << input_image[y * WIDTH + x] << "\t";
        }

        std::cout << std::endl;
    }

    // --------------------------------------------------
    // Display Output Image
    // --------------------------------------------------

    std::cout << "\nOutput Image\n\n";

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            std::cout << output_image[y * WIDTH + x] << "\t";
        }

        std::cout << std::endl;
    }

    // --------------------------------------------------
    // Free GPU Memory
    // --------------------------------------------------

    MemoryManager::freeMemory(gpu_input);
    MemoryManager::freeMemory(gpu_output);

    return 0;
}
