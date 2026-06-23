#pragma once

// Maximum cameras supported by system
constexpr int MAX_CAMERAS = 6;

// SIFT configuration
constexpr int NUM_OCTAVES = 4;
constexpr int SCALES_PER_OCTAVE = 8;

// Validated thresholds
constexpr float CONTRAST_THRESHOLD = 1.0f;

constexpr float RANSAC_THRESHOLD = 5.0f;
constexpr int RANSAC_ITERATIONS = 2000;
