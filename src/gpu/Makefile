# =====================================================
# Makefile for CUDA Gaussian Pyramid
# =====================================================

# Compiler and flags
NVCC = nvcc
CUDA_ARCH = -arch=sm_75  # RTX series; change to sm_70 for V100, sm_86 for RTX 3080
CUDA_FLAGS = $(CUDA_ARCH) -O3 -std=c++14
CXX_FLAGS = -std=c++14

# Source files
SOURCES = main.cu cuda_kernels.cu gaussian_pyramid.cu image_utils.cu
HEADERS = cuda_kernels.cuh image_utils.cuh
OBJECTS = $(SOURCES:.cu=.o)
TARGET = gaussian_pyramid

# =====================================================
# BUILD RULES
# =====================================================

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Linking..."
	$(NVCC) $(CUDA_FLAGS) -o $@ $(OBJECTS) -lcudart
	@echo "Build complete: $(TARGET)"

%.o: %.cu $(HEADERS)
	@echo "Compiling $<..."
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

# =====================================================
# UTILITIES
# =====================================================

clean:
	@echo "Cleaning..."
	rm -f $(OBJECTS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

# Display architecture
info:
	@echo "CUDA Architecture: $(CUDA_ARCH)"
	@echo "CUDA Flags: $(CUDA_FLAGS)"
	@echo "Source files: $(SOURCES)"

# =====================================================
# ARCHITECTURE SELECTION
# =====================================================
# Use: make CUDA_ARCH="-arch=sm_70" for V100
#      make CUDA_ARCH="-arch=sm_75" for RTX 2070/2080 (default)
#      make CUDA_ARCH="-arch=sm_86" for RTX 3080/A100
#      make CUDA_ARCH="-arch=sm_70,sm_75" for multiple

.PHONY: all clean run info
