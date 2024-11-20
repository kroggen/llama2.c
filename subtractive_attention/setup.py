from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
   
setup(
    name='subtractive_attention_cuda',
    ext_modules=[
        CUDAExtension(
            name='subtractive_attention_cuda',
            sources=['subtractive_attention.cpp', 'subtractive_attention_kernel.cu'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
