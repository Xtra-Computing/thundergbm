//
// Created by jiashuai on 18-1-19.
//

#ifndef THUNDERGBM_DEVICE_LAMBDA_H
#define THUNDERGBM_DEVICE_LAMBDA_H

#include "thundergbm/thundergbm.h"
#include "thundergbm/clion_cuda.h"

template<typename L>
__global__ void lambda_kernel(size_t len, L lambda) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        lambda(i);
    }
}

template<typename L>
__global__ void lambda_2d_sparse_kernel(const int *len2, L lambda) {
    int i = blockIdx.x;
    int begin = len2[i];
    int end = len2[i + 1];
    for (int j = begin + blockIdx.y * blockDim.x + threadIdx.x; j < end; j += blockDim.x * gridDim.y) {
        lambda(i, j);
    }
}

///p100 has 56 MPs, using 32*56 thread blocks
template<typename L>
void device_loop(int len, L lambda, unsigned int NUM_BLOCK = 32 *56, unsigned int BLOCK_SIZE=512) {
    if (len > 0) {
        lambda_kernel << < NUM_BLOCK, BLOCK_SIZE >> > (len, lambda);
        CUDA_CHECK(cudaPeekAtLastError());
    }
}


template<typename L>
void device_loop_2d(int len1, const int *len2, L lambda, unsigned int NUM_BLOCK = 32 * 56,
                    unsigned int BLOCK_SIZE = 256) {
    if (len1 > 0) {
        lambda_2d_sparse_kernel << < dim3(len1, NUM_BLOCK), BLOCK_SIZE >> > (len2, lambda);
        CUDA_CHECK(cudaPeekAtLastError());
    }
}

template<typename L>
__global__ void lambda_2d_sparse_kernel_mod(const int mod_val, const int *len2, L lambda) {
    int i = blockIdx.x;
    int begin = len2[i%mod_val];
    int end = len2[i%mod_val + 1];
    for (int j = begin + blockIdx.y * blockDim.x + threadIdx.x; j < end; j += blockDim.x * gridDim.y) {
        lambda(i, j);
    }
}

template<typename L>
void device_loop_2d_mod(int len1, int mod_val, const int *len2, L lambda, unsigned int NUM_BLOCK = 32 * 56,
                    unsigned int BLOCK_SIZE = 256) {
    if (len1 > 0) {
        lambda_2d_sparse_kernel_mod << < dim3(len1, NUM_BLOCK), BLOCK_SIZE >> > (mod_val, len2, lambda);
        CUDA_CHECK(cudaPeekAtLastError());
    }
}

//template<typename L>
//__global__ void lambda_2d_sparse_kernel_zero(const int *len2, L lambda) {
//    int i = blockIdx.x;
//    int len = len2[i + 1] - len2[i];
//    for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < len; j += blockDim.x * gridDim.y) {
//        lambda(i, j);
//    }
//}
//
//template<typename L>
//void device_loop_2d_zero(int len1, const int *len2, L lambda, unsigned int NUM_BLOCK = 32 * 56,
//                    unsigned int BLOCK_SIZE = 256) {
//    if (len1 > 0) {
//        lambda_2d_sparse_kernel_zero << < dim3(len1, NUM_BLOCK), BLOCK_SIZE >> > (len2, lambda);
//        CUDA_CHECK(cudaPeekAtLastError());
//    }
//}
#endif //THUNDERGBM_DEVICE_LAMBDA_H
