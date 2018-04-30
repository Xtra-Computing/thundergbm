//
// Created by jiashuai on 17-9-16.
//

#ifndef THUNDERSVM_SYNCMEM_H
#define THUNDERSVM_SYNCMEM_H
#include <cstddef>
#include <stdio.h>
#include <string.h>
#include <new>
#include <iostream>
#include <stdlib.h>
#include "cuda_runtime_api.h"

using std::cout;

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error == cudaErrorMemoryAllocation) throw std::bad_alloc(); \
    if(error != cudaSuccess)cout << " " << cudaGetErrorString(error); \
  } while (0)

    inline void malloc_host(void **ptr, size_t size) {
        CUDA_CHECK(cudaMallocHost(ptr, size));
    }

    inline void free_host(void *ptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }

    inline void device_mem_copy(void *dst, const void *src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
    }

    /**
     * @brief Auto-synced memory for CPU and GPU
     */
    class SyncMem {
    public:
        SyncMem();

        /**
         * create a piece of synced memory with given size. The GPU/CPU memory will not be allocated immediately, but
         * allocated when it is used at first time.
         * @param size the size of memory (in Bytes)
         */
        explicit SyncMem(size_t size);

        ~SyncMem();

        ///return raw host pointer
        void *host_data();

        ///return raw device pointer
        void *device_data();

        /**
         * set host data pointer to another host pointer, and its memory will not be managed by this class
         * @param data another host pointer
         */
        void set_host_data(void *data);

        /**
         * set device data pointer to another device pointer, and its memory will not be managed by this class
         * @param data another device pointer
         */
        void set_device_data(void *data);

        ///transfer data to host
        void to_host();

        ///transfer data to device
        void to_device();

        ///return the size of memory
        size_t size() const;

        ///to determine the where the newest data locates in
        enum HEAD {
            HOST, DEVICE, UNINITIALIZED
        };

        HEAD head() const;

        static size_t get_total_memory_size() { return total_memory_size; }

        
    private:
        void *device_ptr;
        void *host_ptr;
        bool own_device_data;
        bool own_host_data;
        size_t size_;
        HEAD head_;
		static size_t total_memory_size;
    };


#endif //THUNDERSVM_SYNCMEM_H
