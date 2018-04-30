//
// Created by jiashuai on 17-9-17.
//
#include "syncarray.h"

template<typename T>
SyncArray<T>::SyncArray(size_t count):mem(new SyncMem(sizeof(T) * count)), size_(count) {

}

template<typename T>
SyncArray<T>::~SyncArray() {
    delete mem;
}

template<typename T>
const T *SyncArray<T>::host_data() const {
    to_host();
    return static_cast<T *>(mem->host_data());
}

template<typename T>
const T *SyncArray<T>::device_data() const {
    to_device();
    return static_cast<T *>(mem->device_data());
}


template<typename T>
T *SyncArray<T>::host_data() {
    to_host();
    return static_cast<T *>(mem->host_data());
}

template<typename T>
T *SyncArray<T>::device_data() {
    to_device();
    return static_cast<T *>(mem->device_data());
}

template<typename T>
void SyncArray<T>::resize(size_t count) {
    delete mem;
    mem = new SyncMem(sizeof(T) * count);
    this->size_ = count;
}

template<typename T>
void SyncArray<T>::copy_from(const T *source, size_t count) {
#ifdef USE_CUDA
    thunder::device_mem_copy(mem->device_data(), source, sizeof(T) * count);
#else
    memcpy(mem->host_data(), source, sizeof(T) * count);
#endif
}


template<typename T>
void SyncArray<T>::copy_from(const SyncArray<T> &source) {
    if(size() != source.size()) cout << "destination and source count doesn't match";
#ifdef USE_CUDA
    copy_from(source.device_data(), source.size());
#else
    copy_from(source.host_data(), source.size());
#endif
}

template<typename T>
void SyncArray<T>::mem_set(const T &value) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaMemset(device_data(), value, mem_size()));
#else
    memset(host_data(), value, mem_size());
#endif
}

template
class SyncArray<int>;

template
class SyncArray<float>;

template
class SyncArray<double>;

template
class SyncArray<unsigned int>;
