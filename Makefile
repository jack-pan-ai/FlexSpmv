################################################################################
# Flexible Spmv Makefile
################################################################################

# CUDA toolkit installation path
CUDA_HOME ?= /usr/local/cuda
BIN_DIR := bin

# NVCC compiler
NVCC := $(CUDA_HOME)/bin/nvcc

# Flags for NVCC
NVCC_FLAGS := -O3 -std=c++17 -arch=sm_70 -lcudart -Werror all-warnings

# Paths to include directories
INCLUDES := -I. -I..
INCLUDES += -I$(CUDA_HOME)/include

# Source files and their corresponding executables
TEST_SOURCE := src/flex_spmv_test.cu
TEST_EXEC := $(BIN_DIR)/flex_spmv_test

DATASET_SOURCE := src/flex_spmv_dataset.cu
DATASET_EXEC := $(BIN_DIR)/flex_spmv_dataset

DATASET_FLAT_SOURCE := src/flex_spmv_dataset_flat.cu
DATASET_FLAT_EXEC := $(BIN_DIR)/flex_spmv_dataset_flat

# Default target
all: $(BIN_DIR) $(TEST_EXEC) $(DATASET_EXEC) $(DATASET_FLAT_EXEC)

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule for test executable
$(TEST_EXEC): $(TEST_SOURCE) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Rule for dataset executable
$(DATASET_EXEC): $(DATASET_SOURCE) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Rule for dataset flat executable
$(DATASET_FLAT_EXEC): $(DATASET_FLAT_SOURCE) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Clean rule
clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean