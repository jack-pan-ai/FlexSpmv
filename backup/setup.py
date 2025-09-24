from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flex_spmv',
    ext_modules=[
        CUDAExtension(
            name='flex_spmv',
            sources=[
                'src/flex_spmv_torch.cu',
                'src/flex_spmv_cuda.cu',
            ],
            include_dirs=[
                'include',
                '.'
            ],
            extra_compile_args={
                # Treat C++ files as CUDA files
                'cxx': ['-O3', '-std=c++17', '-x cu'],
                # 'extra_cflags': ["-g", "-O0"],  # -G enables device debug info
                # 'extra_cuda_cflags': ["-G", "-g", "-O0"],  # -G enables device debug info
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-arch=sm_70',  # Changed from sm_80 to sm_70 for better compatibility
                    '-lcudart',
                    '--expt-relaxed-constexpr',  # Add this flag to fix potential __clz errors
                    '--expt-extended-lambda',    # Add this flag for lambda support
                    '--extended-lambda',         # Additional flag for lambda support
                    '-Xcompiler',
                    '-fPIC',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__'
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
