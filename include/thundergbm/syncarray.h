//
// Created by jiashuai on 17-9-17.
//

#ifndef THUNDERGBM_SYNCDATA_H
#define THUNDERGBM_SYNCDATA_H

#include "thundergbm/util/log.h"
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
        CHECK_GT(size_, 0);
        mem->to_host();
    }

    void to_device() const {
        CHECK_GT(size_, 0);
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
        if (get_owner_id() == source.get_owner_id())
            copy_from(source.device_data(), source.size());
        else
            CUDA_CHECK(cudaMemcpyPeer(mem->device_data(), get_owner_id(), source.device_data(), source.get_owner_id(),
                                      source.mem_size()));
#else
        copy_from(source.host_data(), source.size());
#endif
    };

    /**
     * resize to a new size. This will also clear all data.
     * @param count
     */
    void resize(size_t count) {
        if(mem != nullptr || mem != NULL) {
            delete mem;
        }
        mem = new SyncMem(sizeof(T) * count);
        this->size_ = count;
    };

    /*
     * resize to a new size. This will not clear the origin data.
     * @param count
     */
    void resize_without_delete(size_t count) {
//        delete mem;
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
        const T *data = host_data();
        for (i = 0; i < size() - 1 && i < el::base::consts::kMaxLogPerContainer - 1; ++i) {
//    for (i = 0; i < size() - 1; ++i) {
            ostream << data[i] << ",";
        }
        ostream << host_data()[i];
        if (size() <= el::base::consts::kMaxLogPerContainer) {
            ostream << "]";
        } else {
            ostream << ", ...(" << size() - el::base::consts::kMaxLogPerContainer << " more)";
        }
    };

    int get_owner_id() const {
        return mem->get_owner_id();
    }

    //move constructor
    SyncArray(SyncArray<T> &&rhs) noexcept  : mem(rhs.mem), size_(rhs.size_) {
        rhs.mem = nullptr;
        rhs.size_ = 0;
    }

    //move assign
    SyncArray &operator=(SyncArray<T> &&rhs) noexcept {
        delete mem;
        mem = rhs.mem;
        size_ = rhs.size_;

        rhs.mem = nullptr;
        rhs.size_ = 0;
        return *this;
    }

    SyncArray(const SyncArray<T> &) = delete;

    SyncArray &operator=(const SyncArray<T> &) = delete;
    

    //new function clear gpu mem
    void clear_device(){
        to_host();
        mem->free_device();
    }

private:
    SyncMem *mem;
    size_t size_;
};

//SyncArray for multiple devices
template<typename T>
class MSyncArray : public vector<SyncArray<T>> {
public:
    explicit MSyncArray(size_t n_device) : base_class(n_device) {};

    explicit MSyncArray(size_t n_device, size_t size) : base_class(n_device) {
        for (int i = 0; i < n_device; ++i) {
            this->at(i) = SyncArray<T>(size);
        }
    };

    MSyncArray() : base_class() {};

    //move constructor and assign
    MSyncArray(MSyncArray<T> &&) = default;

    MSyncArray &operator=(MSyncArray<T> &&) = default;

    MSyncArray(const MSyncArray<T> &) = delete;

    MSyncArray &operator=(const MSyncArray<T> &) = delete;

private:
    typedef vector<SyncArray<T>> base_class;
};

#endif //THUNDERGBM_SYNCDATA_H
