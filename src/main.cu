#include <iostream>

#include "../include/config.h"
#include "../include/camera_manager.h"

int main()
{
    CameraManager manager;

    manager.active_cameras = 2;

    std::cout << "CUDA SIFT VR Stitcher\n";

    std::cout << "Maximum Cameras Supported: "
              << MAX_CAMERAS
              << std::endl;

    std::cout << "Currently Active Cameras: "
              << manager.active_cameras
              << std::endl;

    return 0;
}
