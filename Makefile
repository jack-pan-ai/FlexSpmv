################################################################################
# Flexible Spmv Makefile
################################################################################

# CUDA toolkit installation path
CUDA_HOME ?= /usr/local/cuda

# NVCC compiler
NVCC := $(CUDA_HOME)/bin/nvcc

# Flags for NVCC
NVCC_FLAGS := -O3 -std=c++17 -arch=sm_70 -lcudart -Werror all-warnings

# Paths to include directories
INCLUDES := -I. -I..
INCLUDES += -I$(CUDA_HOME)/include

# Source files and their corresponding executables
TEST_SOURCE := src/flex_spmv_test.cu
TEST_EXEC := src/flex_spmv_test

DATASET_SOURCE := src/flex_spmv_dataset.cu
DATASET_EXEC := src/flex_spmv_dataset

# Default target
all: $(TEST_EXEC) $(DATASET_EXEC)

# Rule for test executable
$(TEST_EXEC): $(TEST_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Rule for dataset executable
$(DATASET_EXEC): $(DATASET_SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Clean rule
clean:
	rm -f $(TEST_EXEC) $(DATASET_EXEC)

.PHONY: all clean 