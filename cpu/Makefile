################################################################################
# CPU OmpMergeSystem Test Makefile
################################################################################

# Compiler
CXX := g++

# Compilation flags
CXX_FLAGS := -O3 -std=c++17 -Wall -Wextra -fopenmp -march=native

# Debug flags (uncomment for debugging)
# CXX_FLAGS += -g -DDEBUG

# Include directories
INCLUDES := -I.

# Output directory
BIN_DIR := bin

# Source and executable
SOURCE := main.cpp
EXEC := $(BIN_DIR)/merged_system

# Header dependencies
HEADER_FILES := utils.h merged_spmv.h

# Default target
all: $(BIN_DIR) $(EXEC)

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile rule
$(EXEC): $(SOURCE) $(HEADER_FILES) | $(BIN_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -o $@ $<

# Clean rule
clean:
	rm -rf $(BIN_DIR)

# Test targets
test: $(EXEC)
	@echo "=== Running basic test ==="
	./$(EXEC) --rows 8 --cols 16 --nnz 32 --seed 123
	@echo ""
	@echo "=== Running test with double precision ==="
	./$(EXEC) --double --rows 16 --cols 32 --nnz 64 --seed 456


# Quick test with minimal output
test-quiet: $(EXEC)
	./$(EXEC) --quiet --rows 4 --cols 8 --nnz 16

# Large test
test-large: $(EXEC)
	@echo "=== Running large test ==="
	./$(EXEC) --rows 100 --cols 200 --nnz 1000 --seed 789

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build the executable"
	@echo "  test        - Run comprehensive tests"
	@echo "  test-quiet  - Run minimal test"
	@echo "  test-large  - Run large-scale test"
	@echo "  clean       - Remove build artifacts"
	@echo "  help        - Show this help message"

.PHONY: all clean test test-quiet test-large help
