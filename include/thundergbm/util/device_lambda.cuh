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
template<unsigned int NUM_BLOCK = 32 * 56, unsigned int BLOCK_SIZE = 512, typename L>
void device_loop(int len, L lambda) {
    if (len > 0) {
        lambda_kernel << < NUM_BLOCK, BLOCK_SIZE >> > (len, lambda);
        CUDA_CHECK(cudaPeekAtLastError());
    }
}


template<unsigned int NUM_BLOCK = 32 * 56, unsigned int BLOCK_SIZE = 512, typename L>
void device_lambda_2d_sparse(int len1, const int *len2, L lambda) {
    if (len1 > 0) {
        lambda_2d_sparse_kernel << < dim3(len1, NUM_BLOCK), BLOCK_SIZE >> > (len2, lambda);
        CUDA_CHECK(cudaPeekAtLastError());
    }
}

#endif //THUNDERGBM_DEVICE_LAMBDA_H
