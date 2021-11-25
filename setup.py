import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.3.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'assanet_version.py')

    setup(
        name='ASSANet',
        version=version,
        description='The official implementation of "ASSANet: An Anisotropic Separable Set Abastraction for Efficient Point Cloud Representation Learning"',
        install_requires=[
            'numpy',
            'torch>=1.1',
            'numba',
            'easydict',
            'pyyaml'
        ],
        author='Guocheng Qian',
        author_email='guocheng.qian@kaust.edu.sa',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',
                ],
            ),
        ],
    )
