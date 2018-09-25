//
// Created by jiashuai on 17-9-17.
//

#ifndef THUNDERGBM_SYNCDATA_H
#define THUNDERGBM_SYNCDATA_H

#include "thundergbm.h"
#include "syncmem.h"

/**
 * @brief Wrapper of SyncMem with a type
 * @tparam T type of element
 */
template<typename T>
class SyncArray : public el::Loggable {
public:
    /**
     * initialize class that can store given count of elements
     * @param count the given count
     */
    explicit SyncArray(size_t count) : mem(new SyncMem(sizeof(T) * count)), size_(count) {
    }

    SyncArray() : mem(nullptr), size_(0) {}

    ~SyncArray() { delete mem; };

    const T *host_data() const {
        to_host();
        return static_cast<T *>(mem->host_data());
    };

    const T *device_data() const {
        to_device();
        return static_cast<T *>(mem->device_data());
    };

    T *host_data() {
        to_host();
        return static_cast<T *>(mem->host_data());
    };

    T *device_data() {
        to_device();
        return static_cast<T *>(mem->device_data());
    };

    T *device_end() {
        return device_data() + size();
    };

    const T *device_end() const {
        return device_data() + size();
    };

    void set_host_data(T *host_ptr) {
        mem->set_host_data(host_ptr);
    }

    void set_device_data(T *device_ptr) {
        mem->set_device_data(device_ptr);
    }

    void to_host() const {
        mem->to_host();
    }

    void to_device() const {
        mem->to_device();
    }

    /**
     * copy device data. This will call to_device() implicitly.
     * @param source source data pointer (data can be on host or device)
     * @param count the count of elements
     */
    void copy_from(const T *source, size_t count) {

#ifdef USE_CUDA
        thunder::device_mem_copy(mem->device_data(), source, sizeof(T) * count);
#else
        memcpy(mem->host_data(), source, sizeof(T) * count);
#endif
    };

    void copy_from(const SyncArray<T> &source) {

        CHECK_EQ(size(), source.size()) << "destination and source count doesn't match";
#ifdef USE_CUDA
        copy_from(source.device_data(), source.size());
#else
        copy_from(source.host_data(), source.size());
#endif
    };

    /**
     * resize to a new size. This will also clear all data.
     * @param count
     */
    void resize(size_t count) {
        delete mem;
        mem = new SyncMem(sizeof(T) * count);
        this->size_ = count;
    };

    size_t mem_size() const {//number of bytes
        return mem->size();
    }

    size_t size() const {//number of values
        return size_;
    }

    SyncMem::HEAD head() const {
        return mem->head();
    }

    void log(el::base::type::ostream_t &ostream) const override {
        int i;
        ostream << "[";
        for (i = 0; i < size() - 1 && i < el::base::consts::kMaxLogPerContainer - 1; ++i) {
//    for (i = 0; i < size() - 1; ++i) {
            ostream << host_data()[i] << ",";
        }
        ostream << host_data()[i];
        if (size() < el::base::consts::kMaxLogPerContainer - 1) {
            ostream << "]";
        } else {
            ostream << "...";
        }
    };

    int get_owner_id() {
        return mem->get_owner_id();
    }

private:

    SyncArray(const SyncArray<T> &);

    SyncArray &operator=(const SyncArray<T> &);

    SyncMem *mem;
    size_t size_;
};

#endif //THUNDERGBM_SYNCDATA_H
