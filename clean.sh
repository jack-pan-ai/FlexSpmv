#!/bin/bash

# Clean JIT compilation artifacts
echo "Cleaning JIT compilation artifacts..."
find . -name "*.so" -type f -delete
find . -name "*.dylib" -type f -delete
find . -name "*.dll" -type f -delete
find . -name "*.pyd" -type f -delete

# Clean Python bytecode
echo "Cleaning Python bytecode..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -type f -delete
find . -name "*.pyo" -type f -delete
find . -name "*.pyd" -type f -delete

# Clean AOT compilation artifacts
echo "Cleaning AOT compilation artifacts..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
find . -name "*.egg" -type f -delete

# Clean CUDA compilation artifacts
echo "Cleaning CUDA compilation artifacts..."
find . -name "*.o" -type f -delete
find . -name "*.obj" -type f -delete
find . -name "*.ptx" -type f -delete
find . -name "*.cubin" -type f -delete
find . -name "*.fatbin" -type f -delete
find . -name "*.ii" -type f -delete
find . -name "*.i" -type f -delete

echo "Cleaning complete!"