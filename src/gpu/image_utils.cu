#include "image_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// =====================================================
// LOAD PPM IMAGE
// =====================================================
float* loadImagePPM(const char* filename, int& width, int& height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return nullptr;
    }
    
    char magic[3];
    fread(magic, 1, 2, file);
    magic[2] = '\0';
    
    if (strcmp(magic, "P5") != 0) {
        printf("Error: Only P5 PPM format supported\n");
        fclose(file);
        return nullptr;
    }
    
    // Skip comments
    int c;
    while ((c = fgetc(file)) == '#') {
        while (fgetc(file) != '\n');
    }
    ungetc(c, file);
    
    int max_val;
    fscanf(file, "%d %d %d", &width, &height, &max_val);
    fgetc(file);  // skip whitespace
    
    int size = width * height;
    unsigned char* data_uint8 = new unsigned char[size];
    float* data_float = new float[size];
    
    fread(data_uint8, 1, size, file);
    fclose(file);
    
    // Convert to float
    for (int i = 0; i < size; i++) {
        data_float[i] = (float)data_uint8[i] / 255.0f;
    }
    
    delete[] data_uint8;
    return data_float;
}

// =====================================================
// SAVE PPM IMAGE
// =====================================================
bool saveImagePPM(const char* filename, float* data, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Cannot create file %s\n", filename);
        return false;
    }
    
    fprintf(file, "P5\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");
    
    int size = width * height;
    unsigned char* data_uint8 = new unsigned char[size];
    
    // Convert from float to uint8
    for (int i = 0; i < size; i++) {
        float val = data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        data_uint8[i] = (unsigned char)(val * 255.0f);
    }
    
    fwrite(data_uint8, 1, size, file);
    fclose(file);
    
    delete[] data_uint8;
    return true;
}

// =====================================================
// CONVERT TO FLOAT32
// =====================================================
void convertToFloat32(
    unsigned char* src,
    float* dst,
    int size
) {
    for (int i = 0; i < size; i++) {
        dst[i] = (float)src[i] / 255.0f;
    }
}

// =====================================================
// CONVERT TO UINT8
// =====================================================
void convertToUint8(
    float* src,
    unsigned char* dst,
    int size
) {
    for (int i = 0; i < size; i++) {
        float val = src[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        dst[i] = (unsigned char)(val * 255.0f);
    }
}

// =====================================================
// UPLOAD TO GPU
// =====================================================
cudaError_t uploadImageGPU(
    float* h_image,
    float** d_image,
    int width,
    int height
) {
    int size = width * height * sizeof(float);
    
    cudaError_t err = cudaMalloc(d_image, size);
    if (err != cudaSuccess) {
        printf("Error allocating GPU memory: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    err = cudaMemcpy(*d_image, h_image, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error uploading to GPU: %s\n", cudaGetErrorString(err));
        cudaFree(*d_image);
        return err;
    }
    
    return cudaSuccess;
}

// =====================================================
// DOWNLOAD FROM GPU
// =====================================================
cudaError_t downloadImageGPU(
    float* d_image,
    float* h_image,
    int width,
    int height
) {
    int size = width * height * sizeof(float);
    
    cudaError_t err = cudaMemcpy(h_image, d_image, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error downloading from GPU: %s\n", cudaGetErrorString(err));
        return err;
    }
    
    return cudaSuccess;
}
