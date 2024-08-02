from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='HCSPMM',
    ext_modules=[
        CUDAExtension('HCSPMM', [
            'hybrid_all.cpp',
            'hybrid_all_kernel.cu',
        ], extra_compile_args=['-lcublas'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
