################################################################################
# Flexible Spmv Makefile
################################################################################

# CUDA toolkit installation path
CUDA_HOME ?= /usr/local/cuda
BIN_DIR := bin

# NVCC compiler
NVCC := $(CUDA_HOME)/bin/nvcc

# Flags for NVCC
NVCC_FLAGS := -O3 -std=c++17 -arch=sm_80 -lcudart -Werror all-warnings
# NVCC_FLAGS += -G -g # for debug

# Paths to include directories
INCLUDES := -I. -I.. -Iinclude
INCLUDES += -I$(CUDA_HOME)/include

# Source files and their corresponding executables
TEST_SOURCE := src/flex_spmv_test.cu
TEST_EXEC := $(BIN_DIR)/flex_spmv_test

DATASET_SOURCE := src/flex_spmv_dataset.cu
DATASET_EXEC := $(BIN_DIR)/flex_spmv_datasets

# SPRINGMASS_SOURCE := src/flex_spmv_springmass.cu
# SPRINGMASS_EXEC := $(BIN_DIR)/flex_spmv_springmass

# Header files that might be included
HEADER_FILES := $(wildcard include/*.cuh) $(wildcard include/*.h) $(wildcard *.h)

# Default target
all: $(BIN_DIR) $(TEST_EXEC) $(DATASET_EXEC) $(SPRINGMASS_EXEC)

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule for test executable with header dependencies
$(TEST_EXEC): $(TEST_SOURCE) $(HEADER_FILES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Rule for dataset executable with header dependencies
$(DATASET_EXEC): $(DATASET_SOURCE) $(HEADER_FILES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Rule for spring mass executable with header dependencies
$(SPRINGMASS_EXEC): $(SPRINGMASS_SOURCE) $(HEADER_FILES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Clean rule
clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean