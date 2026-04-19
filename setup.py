import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CustomBuildExt(build_ext):
    def run(self):
        try:
            subprocess.check_output(['nvcc', '--version'])
            print("NVIDIA CUDA Toolkit found. Compiling with GPU support.")
            self.build_cuda_module = True
        except FileNotFoundError:
            print("CUDA not found, compiling CPU-only module.")
            self.build_cuda_module = False
        
        self.run_setup_with_extensions()

    def run_setup_with_extensions(self):
        # Create a list of extensions to build
        extensions = []
        if self.build_cuda_module:
            cuda_ext = Extension(
                'my_nn_module',
                sources=['NN.cpp', 'kernels.cu'],
                extra_compile_args={'cxx': ['-std=c++17', '-fPIC'],
                                    'nvcc': ['-std=c++17', '-O3', '--relocatable-device-code=true', '-arch=sm_86']},
                include_dirs=[
                    os.path.join(sys.prefix, 'include'),
                    '/usr/local/cuda/include'
                ],
                library_dirs=['/usr/local/cuda/lib64'],
                runtime_library_dirs=['/usr/local/cuda/lib64'],
                libraries=['cudart'],
            )
            extensions.append(cuda_ext)
        else:
            cpu_ext = Extension(
                'my_nn_module',
                sources=['NN.cpp'],
                extra_compile_args=['-std=c++17', '-fPIC'],
                include_dirs=[os.path.join(sys.prefix, 'include')]
            )
            extensions.append(cpu_ext)

        self.distribution.ext_modules = extensions
        build_ext.run(self)

setup(
    name='my_nn_module',
    version='0.1.3',
    cmdclass={'build_ext': CustomBuildExt},
    ext_modules=[]
)