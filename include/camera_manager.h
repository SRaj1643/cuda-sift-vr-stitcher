#pragma once

#include <vector>
#include <string>

#define MAX_CAMERAS 6

class CameraManager
{
public:

    int active_cameras = 2;

    std::vector<std::string> image_paths;

    CameraManager()
    {
        image_paths.resize(MAX_CAMERAS);
    }
};
