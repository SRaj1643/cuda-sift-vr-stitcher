#pragma once

#include "camera.h"
#include "config.h"

struct CameraManager
{
    int active_cameras;

    CameraData cameras[MAX_CAMERAS];

    CameraManager()
    {
        active_cameras = 0;
    }
};
