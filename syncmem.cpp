//
// Created by jiashuai on 17-9-16.
//

#include "syncmem.h"

    size_t SyncMem::total_memory_size = 0;
    SyncMem::SyncMem() : device_ptr(nullptr), host_ptr(nullptr), size_(0), head_(UNINITIALIZED), own_device_data(false),
                         own_host_data(false) {

    }

    SyncMem::SyncMem(size_t size) : device_ptr(nullptr), host_ptr(nullptr), size_(size), head_(UNINITIALIZED),
                                    own_device_data(false), own_host_data(false) {

    }

    SyncMem::~SyncMem() {
        if (this->head_ != UNINITIALIZED) {
            this->head_ = UNINITIALIZED;
            if (own_host_data || own_device_data) total_memory_size -= size_;
            if (host_ptr && own_host_data) {
                free_host(host_ptr);
                host_ptr = nullptr;
            }
            if (device_ptr && own_device_data) {
                CUDA_CHECK(cudaFree(device_ptr));
                device_ptr = nullptr;
            }
        }
    }

    void *SyncMem::host_data() {
        to_host();
        return host_ptr;
    }

    void *SyncMem::device_data() {
        to_device();
        return device_ptr;
    }

    size_t SyncMem::size() const {
        return size_;
    }

    SyncMem::HEAD SyncMem::head() const {
        return head_;
    }

    void SyncMem::to_host() {
        switch (head_) {
            case UNINITIALIZED:
                malloc_host(&host_ptr, size_);
                memset(host_ptr, 0, size_);
                head_ = HOST;
                own_host_data = true;
                total_memory_size += size_;
                break;
            case DEVICE:
                if (nullptr == host_ptr) {
                    CUDA_CHECK(cudaMallocHost(&host_ptr, size_));
                    CUDA_CHECK(cudaMemset(host_ptr, 0, size_));
                    own_host_data = true;
                }
                CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, size_, cudaMemcpyDeviceToHost));
                head_ = HOST;
                break;
            case HOST:;
        }
    }

    void SyncMem::to_device() {
        switch (head_) {
            case UNINITIALIZED:
                CUDA_CHECK(cudaMalloc(&device_ptr, size_));
                CUDA_CHECK(cudaMemset(device_ptr, 0, size_));
                head_ = DEVICE;
                own_device_data = true;
                total_memory_size += size_;
                break;
            case HOST:
                if (nullptr == device_ptr) {
                    CUDA_CHECK(cudaMalloc(&device_ptr, size_));
                    CUDA_CHECK(cudaMemset(device_ptr, 0, size_));
                    own_device_data = true;
                }
                CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, size_, cudaMemcpyHostToDevice));
                head_ = DEVICE;
                break;
            case DEVICE:;
        }
    }

    void SyncMem::set_host_data(void *data) {
        if(data == NULL){cout << "oh shit\n"; exit(0);};
        if (own_host_data) {
            free_host(host_ptr);
            total_memory_size -= size_;
        }
        host_ptr = data;
        own_host_data = false;
        head_ = HEAD::HOST;
    }

    void SyncMem::set_device_data(void *data) {
    	if(data == NULL){cout << "oh shit\n"; exit(0);};
        if (own_device_data) {
            CUDA_CHECK(cudaFree(device_data()));
            total_memory_size -= size_;
        }
        device_ptr = data;
        own_device_data = false;
        head_ = HEAD::DEVICE;
    }

