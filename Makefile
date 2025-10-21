################################################################################
# Flexible Spmv Combined Makefile (CUDA + CPU)
################################################################################

# CUDA toolkit installation path
CUDA_HOME ?= /usr/local/cuda
BIN_DIR := bin

# Compilers
NVCC := $(CUDA_HOME)/bin/nvcc
CXX := g++

# Flags for NVCC
NVCC_FLAGS := -O3 -std=c++17 -arch=sm_70 -lcudart -Werror all-warnings --extended-lambda
# NVCC_FLAGS += -G -g # for debug

# Flags for CPU compilation
CXX_FLAGS := -O3 -std=c++17 -Wall -Wextra -fopenmp -march=native -Wunused-parameter
# CXX_FLAGS += -g -DDEBUG # for debug

# Paths to include directories
INCLUDES := -I. -I.. -Iinclude
INCLUDES += -I$(CUDA_HOME)/include

# CUDA Source files and executables
# FLEX_SOURCE := src/easier_module_copy.cu
# FLEX_EXEC := $(BIN_DIR)/easier_module_copy
# FLEX_SOURCE := src/easier_module.cu
# FLEX_EXEC := $(BIN_DIR)/easier_module
# # (optional) used for testing full
# FLEX_SOURCE := src/flex_spmv_full_red.cu
# FLEX_EXEC := $(BIN_DIR)/flex_spmv_full_red

# # # (optional) used for testing full
# FLEX_SOURCE := src/flex_spmv_full_agg.cu
# FLEX_EXEC := $(BIN_DIR)/flex_spmv_full_agg

# # (optional) used for testing map
# FLEX_SOURCE := src/flex_spmv_map.cu
# FLEX_EXEC := $(BIN_DIR)/flex_spmv_map

# CPU Source files and executables
CPU_SOURCE := main.cpp
CPU_EXEC := $(BIN_DIR)/merged_system_cpu
# (optional) used for testing map
# CPU_SOURCE := mainMap.cpp
# CPU_EXEC := $(BIN_DIR)/merged_system_cpu_map

# Header files that might be included
HEADER_FILES := $(wildcard include/*.cuh) $(wildcard include/*.h) $(wildcard *.h)
CPU_HEADER_FILES := merged_spmv.h

# Default target - build both CUDA and CPU versions
all: $(BIN_DIR) $(FLEX_EXEC) $(CPU_EXEC)

# CUDA-only target
cuda: $(BIN_DIR) $(FLEX_EXEC)

# CPU-only target  
cpu: $(BIN_DIR) $(CPU_EXEC)

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule for CUDA executable with header dependencies
$(FLEX_EXEC): $(FLEX_SOURCE) $(HEADER_FILES) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $<

# Rule for CPU executable with header dependencies
$(CPU_EXEC): $(CPU_SOURCE) $(CPU_HEADER_FILES) | $(BIN_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -o $@ $<

# Clean rule
clean:
	rm -rf $(BIN_DIR)

# Test commands (examples)
test-cpu:
	./$(CPU_EXEC) --rows 100 --cols 200 --nnz 1000 --seed 789

.PHONY: all cuda cpu clean test-cpu