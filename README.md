# Pytorch 2.0.1 - tie4

## Usefull tutorial: https://www.youtube.com/watch?v=AIkvOtHJZeo

# Requirments
- Python 3.8 or later (for Linux, Python 3.8.1+ is needed)
- A C++17 compatible compiler
- NVIDIA CUDA 11.7 
- NVIDIA cuDNN v8 is mandatory! 
- torch-2.1.0a0+git0147646

# Compile
- To deactivate: conda deactivate
- Remove pytorch from pip: pip3.9 uninstall torch
- Install anaconda
  1. wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
  2. bash Anaconda3-2023.03-1-Linux-x86_64.sh
  3. I HAVE INSTALL Anacoda to /tmp/manospavl/anaconda or /spare/anaconda3
    - logout and login
  4.  Create anaconda env: conda create --name pytorch-dev python=3.9
      Activate anaconda: conda activate pytorch-dev 
    Every time I should see (pytorch-dev) manospavl@tie1:pytorch$
    5. conda install cmake ninja mkl mkl-include
  6. conda install -c pytorch magma-cuda117 --> cuda 11.7
  7. git clone --recursive https://github.com/pytorch/pytorch
    -- git checkout v2.0.1
  8.  cd pytorch
  9.  pip3.9 install --user -r requirements.txt
  10. git submodule sync; git submodule update --init --recursive
  11. export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
  12. echo ${CMAKE_PREFIX_PATH} --> /tmp/manospavl/anaconda/envs/pytorch-dev

  13. python3.9 setup.py build --cmake-only --> Stop once cmake terminates. Leave users a chance to adjust build options.
    -- If you see errors remove build/
  14. - cmake -DCUDA_SEPARABLE_COMPILATION=ON -DUSE_CUDNN=OFF -DUSE_EXPERIMENTAL_CUDNN_V8_API=OFF -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_SKIP_INSTALL_RPATH=ON -DCMAKE_SKIP_RPATH=ON build 
  15. According to setup.py then I have to run python setup.py install to build!! This generates  libtorch.so inside /spare/manospavl/pytorch/torch/lib
    - python3.9 setup.py install
  16. If I do ninja -j 32 inside build it generated libtorch inside build/lib 
  17. To clean: python setup.py clean

  # Basic Compilation Steps: 
  - python3.9 setup.py clean
  - python3.9 setup.py build --cmake-only
  - cmake -DCUDA_SEPARABLE_COMPILATION=ON -DUSE_CUDNN=OFF -DUSE_EXPERIMENTAL_CUDNN_V8_API=OFF -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_SKIP_INSTALL_RPATH=ON -DCMAKE_SKIP_RPATH=ON build 
  - python3.9 setup.py install
  # Compile Pytorch with debug symbols
  - export DEBUG=1 and follow the steps above!
  
# PTX
- According to my modifications some ptx are extracted to build.
- Other are in the form of .o so they need cu_extract:  /spare/manospavl/pytorch/build/caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/cuda
- And finally get ptx from libtorch_cuda.so


# STATIC LINKING
- modified:   CMakeLists.txt
- modified:   aten/src/ATen/CMakeLists.txt
- modified:   binaries/CMakeLists.txt
- modified:   caffe2/CMakeLists.txt
- modified:   cmake/Modules_CUDA_fix/upstream/FindCUDA.cmake
- modified:   cmake/public/cuda.cmake

For more details see static_pytorch.info
 - Disable NCCL 
 - Enable CUDA_RESOLVE_DEVICE_SYMBOLS ON from cmake


##cuDNN
cudnn-linux-x86_64-8.9.2.26_cuda11-archive.tar.xz or cudnn-local-repo-rhel7-8.9.2.26-1.0-1.x86_64.rpm

## CUBLAS
I have changed cmake/Modules_CUDA_fix/upstream/FindCUDA.cmake 
 -   Comment:      #unset(CUDA_cublas_LIBRARY CACHE), #unset(CUDA_cublasLt_LIBRARY CACHE) 
     Add this:      unset(CUBLAS_STATIC_LIBRARY CACHE), unset(CUBLASLT_STATIC_LIBRARY CACHE)
 - Then i modify the following 
  if (CUDA_BUILD_EMULATION)
    set(CUDA_CUFFT_LIBRARIES ${CUDA_cufftemu_LIBRARY})
    set(CUDA_CUBLAS_LIBRARIES ${CUDA_cublasemu_LIBRARY})
  else()
    message(!!!!!CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR}")
    find_library(CULIBOS_STATIC_LIBRARY culibos PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    find_library(CUBLASLT_STATIC_LIBRARY cublasLt_static PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    find_library(CUBLAS_STATIC_LIBRARY cublas_static PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    set(CUDA_CUBLAS_LIBRARIES ${CUBLAS_STATIC_LIBRARY} ${CUBLASLT_STATIC_LIBRARY} ${CULIBOS_STATIC_LIBRARY} CACHE FILEPATH "" FORCE)
    set(CUDA_CUFFT_LIBRARIES ${CUDA_cufft_LIBRARY})
    #set(CUDA_CUBLAS_LIBRARIES ${CUDA_cublas_LIBRARY} ${CUDA_cublas_device_LIBRARY} ${CUDA_cublasLt_LIBRARY})
  endif()

- And the following 
  macro(CUDA_ADD_CUBLAS_TO_TARGET target)
    if (CUDA_BUILD_EMULATION)
      target_link_libraries(${target} ${CUDA_LINK_LIBRARIES_KEYWORD} ${CUDA_cublasemu_LIBRARY})
    else()
      #target_link_libraries(${target} ${CUDA_LINK_LIBRARIES_KEYWORD} ${CUDA_cublas_LIBRARY} ${CUDA_cublas_device_LIBRARY} ${CUDA_cublasLt_LIBRARY})
      target_link_libraries(${target} ${CUDA_LINK_LIBRARIES_KEYWORD} ${CUBLAS_STATIC_LIBRARY} ${CUBLASLT_STATIC_LIBRARY} Threads::Threads dl)
    endif()
  endmacro()
- If you see this /usr/lib64/libcublasLt_static.a fix that:
  sudo ln -s /usr/local/cuda/lib64/libcublasLt_static.a /usr/lib64/libcublasLt_static.a
  sudo ln -s /usr/local/cuda/lib64/libculibos.a /usr/lib64/libculibos.a
  sudo ln -s /usr/local/cuda/lib64/libcublas_static.a /usr/lib64/libcublas_static.a


# To check if pytorch is installed use:  pip3 show torch
  DO NOT install requirmens !! If it requires torchvision install only this!
  (pytorch-dev) manospavl@tie1:build$ pip3 show torch 
  ## Correct output:

  Name: torch
  Version: 2.0.0a0+gite9ebda2
  Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
  Home-page: https://pytorch.org/
  Author: PyTorch Team
  Author-email: packages@pytorch.org
  License: BSD-3
  Location: /spare/manospavl/pytorch
  Editable project location: /spare/manospavl/pytorch
  Requires: filelock, jinja2, networkx, sympy, typing-extensions
  Required-by: torchvision, triton

## Wrong Output
  - When i install the requirments in mnist python version the pip3 show torch shows:
  Name: torch
  Version: 2.0.1
  Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
  Home-page: https://pytorch.org/
  Author: PyTorch Team
  Author-email: packages@pytorch.org
  License: BSD-3
  Location: /tmp/manospavl/anaconda/envs/pytorch-dev/lib/python3.9/site-packages
  Editable project location: /tmp/manospavl/anaconda/envs/pytorch-dev/lib/python3.9/site-packages
  Requires: filelock, jinja2, networkx, nvidia-cublas-cu11, nvidia-cuda-cupti-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-runtime-cu11, nvidia-cudnn-cu11, nvidia-cufft-cu11, nvidia-curand-cu11, nvidia-cusolver-cu11, nvidia-cusparse-cu11, nvidia-nccl-cu11, nvidia-nvtx-cu11, sympy, triton, typing-extensions
  Required-by: torchvision, triton

  - Which generates the error with GET!! Do pip3.9 uninstall torch

 Run examples
  - Then cpp/mnist with cmake -DCMAKE_PREFIX_PATH=/spare/manospavl/pytorch -DCMAKE_SKIP_INSTALL_RPATH=ON -DCMAKE_SKIP_RPATH=ON ../
  runs successfully. If I do the above and do not recompile mnist cpp version it runs!!

# Example Mnist cpp: 
  cmake -DCMAKE_PREFIX_PATH=/spare/manospavl/pytorch/ ../
# Run 
  export CUDA_MODULE_LOADING=LAZY; export CUDA_FORCE_PTX_JIT=1; LD_PRELOAD=./../../../../../Interposer/libmylib_nat.so taskset -c 9 ./mnist &> mnist_res_2.txt

# Pytorch .so location: 
ldd /spare/manospavl/pytorch/torch/lib/libtorch.so

##################################################################################
#### Install Python 3.9 ###
wget https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz

tar xvf Python-3.9.16.tgz
cd Python-3.9.16/
./configure --enable-optimizations
sudo make altinstall
sudo ln -sfn /usr/local/bin/python3.9 /usr/bin/python3.9
sudo ln -sfn /usr/local/bin/pip3.9 /usr/bin/pip3.9

########## Conda usefull cmds #########
conda env list 
conda env remove --name 
##################################################################################
## Conda setup ##
WARNING:
    You currently have a PYTHONPATH environment variable set. This may cause
    unexpected behavior when running the Python interpreter in Anaconda3.
    For best results, please verify that your PYTHONPATH only points to
    directories of packages that are compatible with the Python interpreter
    in Anaconda3: /tmp/manospavl/anaconda
Do you wish the installer to initialize Anaconda3
by running conda init? [yes|no]
[no] >>> yes
no change     /tmp/manospavl/anaconda/condabin/conda
no change     /tmp/manospavl/anaconda/bin/conda
no change     /tmp/manospavl/anaconda/bin/conda-env
no change     /tmp/manospavl/anaconda/bin/activate
no change     /tmp/manospavl/anaconda/bin/deactivate
no change     /tmp/manospavl/anaconda/etc/profile.d/conda.sh
no change     /tmp/manospavl/anaconda/etc/fish/conf.d/conda.fish
no change     /tmp/manospavl/anaconda/shell/condabin/Conda.psm1
no change     /tmp/manospavl/anaconda/shell/condabin/conda-hook.ps1
no change     /tmp/manospavl/anaconda/lib/python3.10/site-packages/xontrib/conda.xsh
no change     /tmp/manospavl/anaconda/etc/profile.d/conda.csh
modified      /home1/public/manospavl/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

If you'd prefer that conda's base environment not be activated on startup, 
   set the auto_activate_base parameter to false: 

conda config --set auto_activate_base false

Thank you for installing Anaconda3!

##################################################################################
## Trouble Shooting ##
[Errno 13] Permission denied: '/usr/local/lib/python3.9/site-packages/test-easy-install-41691.write-test'

The installation directory you specified (via --install-dir, --prefix, or
the distutils default setting) was:
/usr/local/lib/python3.9/site-packages/

Solution : sudo chmod 777 /usr/local/lib/python3.9/site-packages/
##################################################################################

## cuDNN v8 ##
CMake Error at cmake/public/cuda.cmake:277 (message):
  PyTorch requires cuDNN 8 and above.
Call Stack (most recent call first):
  cmake/Dependencies.cmake:43 (include)
  CMakeLists.txt:717 (include)


#########################################
Search in /spare/manospavl/pytorch/torch/lib
grep -nr "_ZN2at6native18elementwise_kernelILi128ELi2EZNS0_15gpu_kernel_implINS0_15CUDAFunctor_addIfEEEEvRNS_18TensorIteratorBaseERKT_EUliE_EEviT1_"

cat pytorch_klist.json | grep "_ZN8epilogue4impl12globalKernelILi8ELi32E6__halfS2_fLb0ELb0EEEviilPT1_18cublasLtEpilogue_tiPT2_lPvlllPT3_lPi" 


