#!/bin/bash
echo "Clean"
python3.9 setup.py clean
echo "Build"
python3.9 setup.py build --cmake-only &> build.txt
echo "Cmake"
cmake -DCUDA_SEPARABLE_COMPILATION=ON -DUSE_CUDNN=OFF -DUSE_EXPERIMENTAL_CUDNN_V8_API=OFF -DCMAKE_CUDA_ARCHITECTURES=80 build  &> cmake.txt
echo "Install"
python3.9 setup.py install &> install.txt
