################################################################################
# Flexible Spmv Makefile
################################################################################

# CUDA toolkit installation path
CUDA_HOME ?= /usr/local/cuda

# NVCC compiler
NVCC := $(CUDA_HOME)/bin/nvcc

# Flags for NVCC
NVCC_FLAGS := -O3 -std=c++17 -arch=sm_70 -lcudart

# Paths to include directories
INCLUDES := -I. -I..
INCLUDES += -I$(CUDA_HOME)/include

# Source files
SOURCES := src/flex_spmv_test.cu

# Output executable
EXECUTABLE := flex_spmv_test

# Rule to build the executable
all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Clean rule
clean:
	rm -f $(EXECUTABLE)

.PHONY: all clean 