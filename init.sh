#!/usr/bin/env bash
# command to install this enviroment: source init.sh

# install anaconda3.
#cd ~/
#wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
#bash Anaconda3-2019.10-Linux-x86_64.sh
# source ~/.bashrc

export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"   # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
module purge
module load cuda/11.1.1
module load gcc

# make sure local cuda version is 11.1
conda deactivate
conda env remove --name assanet
conda create -n assanet -y python=3.7 numpy=1.20 numba # do not install high version of numpy as it conflicts with numba
conda activate assanet
# make sure pytorch version >=1.4.0
# NOTE: 'nvidia' channel is required for cudatoolkit 11.1
conda install -y pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia

# install useful modules
pip install tensorboard termcolor tensorboard h5py easydict tqdm pyyaml wandb sklearn multimethod

# compile custom operators
cd ops/cpp_wrappers
sh compile_wrappers.sh
cd ../pt_custom_ops
python setup.py install --user
cd ../..

python setup.py develop
