//
// Created by jiashuai on 17-9-16.
//

#include <thundergbm/syncmem.h>

namespace thunder {
    Allocator SyncMem::cub_allocator(8, 3, 10, CachingDeviceAllocator::INVALID_SIZE, true,false);

    SyncMem::SyncMem() : SyncMem(0) {}

    SyncMem::SyncMem(size_t size) : device_ptr(nullptr), host_ptr(nullptr), size_(size), head_(UNINITIALIZED),
                                    own_device_data(false), own_host_data(false) {
#ifdef USE_CUDA
        CUDA_CHECK(cudaGetDevice(&device_id));
#endif
    }

    SyncMem::~SyncMem() {
        this->head_ = UNINITIALIZED;
        if (host_ptr && own_host_data) {
            free_host(host_ptr);
            host_ptr = nullptr;
        }
#ifdef USE_CUDA
        DO_ON_DEVICE(device_id, {
            if (device_ptr && own_device_data) {
//                CUDA_CHECK(cudaFree(device_ptr));
                cub_allocator.DeviceFree(device_ptr);
                device_ptr = nullptr;
            }
        });
#endif
    }

    void *SyncMem::host_data() {
        to_host();
        return host_ptr;
    }

    void *SyncMem::device_data() {
#ifdef USE_CUDA
        to_device();
#else
        NO_GPU;
#endif
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
                break;
            case DEVICE:
#ifdef USE_CUDA
                DO_ON_DEVICE(device_id, {
                    if (nullptr == host_ptr) {
                        CUDA_CHECK(cudaHostAlloc(&host_ptr, size_, cudaHostAllocPortable));
//                        malloc_host(&host_ptr, size_, cudaHostAllocPortable);
                        CUDA_CHECK(cudaMemset(host_ptr, 0, size_));
                        own_host_data = true;
                    }
                    CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, size_, cudaMemcpyDeviceToHost));
                    head_ = HOST;
                });
#else
                NO_GPU;
#endif
                break;
            case HOST:;
        }
    }

    void SyncMem::to_device() {
#ifdef USE_CUDA
//        DO_ON_DEVICE(device_id, {
        switch (head_) {
            case UNINITIALIZED:
//                    CUDA_CHECK(cudaMalloc(&device_ptr, size_));
                CUDA_CHECK(cub_allocator.DeviceAllocate(&device_ptr, size_));
                CUDA_CHECK(cudaMemset(device_ptr, 0, size_));
                head_ = DEVICE;
                own_device_data = true;
                break;
            case HOST:
                if (nullptr == device_ptr) {
//                        CUDA_CHECK(cudaMalloc(&device_ptr, size_));
                    CUDA_CHECK(cub_allocator.DeviceAllocate(&device_ptr, size_));
                    CUDA_CHECK(cudaMemset(device_ptr, 0, size_));
                    own_device_data = true;
                }
                CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, size_, cudaMemcpyHostToDevice));
                head_ = DEVICE;
                break;
            case DEVICE:;
        }
//        });
#else
        NO_GPU;
#endif
    }

    void SyncMem::set_host_data(void *data) {
        CHECK_NOTNULL(data);
        if (own_host_data) {
            free_host(host_ptr);
        }
        host_ptr = data;
        own_host_data = false;
        head_ = HEAD::HOST;
    }

    void SyncMem::set_device_data(void *data) {
#ifdef USE_CUDA
        DO_ON_DEVICE(device_id, {
            CHECK_NOTNULL(data);
            if (own_device_data) {
                CUDA_CHECK(cudaFree(device_data()));
            }
            device_ptr = data;
            own_device_data = false;
            head_ = HEAD::DEVICE;
        });
#else
        NO_GPU;
#endif
    }

    cudaError_t Allocator::DeviceAllocate(int device, void **d_ptr, size_t bytes, cudaStream_t active_stream) {
        *d_ptr = NULL;
        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        cudaError_t error = cudaSuccess;

        if (device == INVALID_DEVICE_ORDINAL) {
            if (CubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
            device = entrypoint_device;
        }

        // Create a block descriptor for the requested allocation
        bool found = false;
        BlockDescriptor search_key(device);
        search_key.associated_stream = active_stream;
        NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

        if (search_key.bin > max_bin) {
            // Bin is greater than our maximum bin: allocate the request
            // exactly and give out-of-bounds bin.  It will not be cached
            // for reuse when returned.
            search_key.bin = max_bin + 1;
            search_key.bytes = bytes;
        } else {
            // Search for a suitable cached allocation: lock

            if (search_key.bin < min_bin) {
                // Bin is less than minimum bin: round up
                search_key.bin = min_bin;
                search_key.bytes = min_bin_bytes;
            }
        }

        mutex.Lock();
        // Iterate through the range of cached blocks on the same device in the same bin
        CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
        while ((block_itr != cached_blocks.end())
               && (block_itr->device == device)
               && (block_itr->bin == search_key.bin)
               && (block_itr->bytes == search_key.bytes)) {
            // To prevent races with reusing blocks returned by the host but still
            // in use by the device, only consider cached blocks that are
            // either (from the active stream) or (from an idle stream)
            if ((active_stream == block_itr->associated_stream) ||
                (cudaEventQuery(block_itr->ready_event) != cudaErrorNotReady)) {
                // Reuse existing cache block.  Insert into live blocks.
                found = true;
                search_key = *block_itr;
                search_key.associated_stream = active_stream;
                live_blocks.insert(search_key);

                // Remove from free blocks
                cached_bytes[device].free -= search_key.bytes;
                cached_bytes[device].live += search_key.bytes;

                if (debug)
                    _CubLog("\tDevice %d reused cached block at %p (%lld bytes) for stream %lld (previously associated with stream %lld).\n",
                            device, search_key.d_ptr, (long long) search_key.bytes,
                            (long long) search_key.associated_stream, (long long) block_itr->associated_stream);

                cached_blocks.erase(block_itr);

                break;
            }
            block_itr++;
        }

        // Done searching: unlock
        mutex.Unlock();

        // Allocate the block if necessary
        if (!found) {
            // Set runtime's current device to specified device (entrypoint may not be set)
            if (device != entrypoint_device) {
                if (CubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
                if (CubDebug(error = cudaSetDevice(device))) return error;
            }

            // Attempt to allocate
            if (CubDebug(error = cudaMalloc(&search_key.d_ptr, search_key.bytes)) == cudaErrorMemoryAllocation) {
                // The allocation attempt failed: free all cached blocks on device and retry
                if (debug)
                    _CubLog("\tDevice %d failed to allocate %lld bytes for stream %lld, retrying after freeing cached allocations",
                            device, (long long) search_key.bytes, (long long) search_key.associated_stream);

                error = cudaSuccess;    // Reset the error we will return
                cudaGetLastError();     // Reset CUDART's error

                // Lock
                mutex.Lock();

                // Iterate the range of free blocks on the same device
                BlockDescriptor free_key(device);
                CachedBlocks::iterator block_itr = cached_blocks.lower_bound(free_key);

                while ((block_itr != cached_blocks.end()) && (block_itr->device == device)) {
                    // No need to worry about synchronization with the device: cudaFree is
                    // blocking and will synchronize across all kernels executing
                    // on the current device

                    // Free device memory and destroy stream event.
                    if (CubDebug(error = cudaFree(block_itr->d_ptr))) break;
                    if (CubDebug(error = cudaEventDestroy(block_itr->ready_event))) break;

                    // Reduce balance and erase entry
                    cached_bytes[device].free -= block_itr->bytes;

                    if (debug)
                        _CubLog("\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                                device, (long long) block_itr->bytes, (long long) cached_blocks.size(),
                                (long long) cached_bytes[device].free, (long long) live_blocks.size(),
                                (long long) cached_bytes[device].live);

                    cached_blocks.erase(block_itr);

                    block_itr++;
                }

                // Unlock
                mutex.Unlock();

                // Return under error
                if (error) return error;

                // Try to allocate again
                if (CubDebug(error = cudaMalloc(&search_key.d_ptr, search_key.bytes))) return error;
            }

            // Create ready event
            if (CubDebug(error = cudaEventCreateWithFlags(&search_key.ready_event, cudaEventDisableTiming)))
                return error;

            // Insert into live blocks
            mutex.Lock();
            live_blocks.insert(search_key);
            cached_bytes[device].live += search_key.bytes;
            mutex.Unlock();

            if (debug)
                _CubLog("\tDevice %d allocated new device block at %p (%lld bytes associated with stream %lld).\n",
                        device, search_key.d_ptr, (long long) search_key.bytes,
                        (long long) search_key.associated_stream);

            // Attempt to revert back to previous device if necessary
            if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device)) {
                if (CubDebug(error = cudaSetDevice(entrypoint_device))) return error;
            }
        }

        // Copy device pointer to output parameter
        *d_ptr = search_key.d_ptr;

        if (debug)
            _CubLog("\t\t%lld available blocks cached (%lld bytes), %lld live blocks outstanding(%lld bytes).\n",
                    (long long) cached_blocks.size(), (long long) cached_bytes[device].free,
                    (long long) live_blocks.size(), (long long) cached_bytes[device].live);

        return error;
    }

    Allocator::Allocator(unsigned int bin_growth, unsigned int min_bin, unsigned int max_bin, size_t max_cached_bytes,
                         bool skip_cleanup, bool debug) : CachingDeviceAllocator(bin_growth, min_bin, max_bin,
                                                                                 max_cached_bytes, skip_cleanup,
                                                                                 debug) {}

    cudaError_t Allocator::DeviceAllocate(void **d_ptr, size_t bytes, cudaStream_t active_stream) {
        return DeviceAllocate(INVALID_DEVICE_ORDINAL, d_ptr, bytes, active_stream);
    }
}
