# Topology Optimization

## Compile SuiteSparse with CUDA
1. setup mkl env:
    ```
    source /opt/intel/mkl/bin/mklvars.sh intel64
    ```
1. clone the library:
    ```
    git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
    ```

2. modify ```SuiteSparse/SuiteSparse_config/SuiteSparse_config.mk``` as follows:

    - Setup CUDA_PATH
    ```
    CUDA_PATH = $(shell which nvcc 2>/dev/null | sed "s/\/bin\/nvcc//")
    ```
    change to
    ```
    CUDA_PATH = /usr/local/cuda-10.1
    ```

    - If your CUDA version >= 10.1, you should change the path of ```libcublas.so```:
    ```
    CUBLAS_LIB    = $(CUDA_PATH)/lib64/libcublas.so
    ```
    change to
    ```
    CUBLAS_LIB = /usr/lib/x86_64-linux-gnu/libcublas.so
    ```

3. compile the library using the following command at the root of SuiteSparse:
    ```
    make library BLAS='-lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -lmkl_blas95_lp64 -liomp5' LAPACK='-lmkl_lapack95_lp64' -j 12
    ```

## Compile ZIRAN with SuiteSparse without root access
```
cmake -DCMAKE_BUILD_TYPE=Release -DSuiteSparse_ROOT=/home/xuan/code/SuiteSparse ..
```
