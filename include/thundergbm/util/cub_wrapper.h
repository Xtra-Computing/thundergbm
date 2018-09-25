//
// Created by ss on 18-5-13.
//

#ifndef THUNDERGBM_CUB_UTIL_H
#define THUNDERGBM_CUB_UTIL_H

#include "thundergbm/thundergbm.h"
//#include <cub/util_allocator.cuh>
//#include <cub/device/device_radix_sort.cuh>
#include "cub/cub.cuh"
#include <thundergbm/syncarray.h>

template<typename T1, typename T2>
void cub_sort_by_key(SyncArray<T1> &keys, SyncArray<T2> &values, bool ascending = true) {
    CHECK_EQ(values.size(), values.size()) << "keys and values must have equal size";
    using namespace cub;
    size_t num_items = keys.size();
    SyncArray<T1> keys2(num_items);
    SyncArray<T2> values2(num_items);
    SyncArray<char> temp_storage;

    DoubleBuffer<T1> d_keys(keys.device_data(), keys2.device_data());
    DoubleBuffer<T2> d_values(values.device_data(), values2.device_data());

    size_t temp_storage_bytes = 0;

    // Initialize device arrays
    if (ascending)
        DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys, d_values, num_items);
    else
        DeviceRadixSort::SortPairsDescending(NULL, temp_storage_bytes, d_keys, d_values, num_items);
    temp_storage.resize(temp_storage_bytes);

    // Run
    if (ascending)
        DeviceRadixSort::SortPairs(temp_storage.device_data(), temp_storage_bytes, d_keys, d_values, num_items);
    else
        DeviceRadixSort::SortPairsDescending(temp_storage.device_data(), temp_storage_bytes, d_keys, d_values,
                                             num_items);

    CUDA_CHECK(
            cudaMemcpy(keys.device_data(), reinterpret_cast<const void *>(d_keys.Current()), sizeof(float) * num_items,
                       cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(values.device_data(), reinterpret_cast<const void *>(d_values.Current()),
                          sizeof(int) * num_items,
                          cudaMemcpyDeviceToDevice));
}

template<typename T1, typename T2>
void cub_seg_sort_by_key(SyncArray<T1> &keys, SyncArray<T2> &values, SyncArray<int> &ptr, bool ascending = true) {
    CHECK_EQ(values.size(), values.size()) << "keys and values must have equal size";
    using namespace cub;
    size_t num_items = keys.size();
    size_t num_segments = ptr.size() - 1;
    SyncArray<T1> keys2(num_items);
    SyncArray<T2> values2(num_items);
    SyncArray<char> temp_storage;

    DoubleBuffer<T1> d_keys(keys.device_data(), keys2.device_data());
    DoubleBuffer<T2> d_values(values.device_data(), values2.device_data());

    size_t temp_storage_bytes = 0;

    // Initialize device arrays
    if (ascending)
        DeviceSegmentedRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys, d_values, num_items, num_segments,
                                            ptr.device_data(), ptr.device_data() + 1);
    else
        DeviceSegmentedRadixSort::SortPairsDescending(NULL, temp_storage_bytes, d_keys, d_values, num_items,
                                                      num_segments,
                                                      ptr.device_data(), ptr.device_data() + 1);
    temp_storage.resize(temp_storage_bytes);

    // Run
    if (ascending)
        DeviceSegmentedRadixSort::SortPairs(temp_storage.device_data(), temp_storage_bytes, d_keys, d_values,
                                            num_items, num_segments, ptr.device_data(),
                                            ptr.device_data() + 1);
    else
        DeviceSegmentedRadixSort::SortPairsDescending(temp_storage.device_data(), temp_storage_bytes, d_keys, d_values,
                                                      num_items, num_segments, ptr.device_data(),
                                                      ptr.device_data() + 1);

    CUDA_CHECK(
            cudaMemcpy(keys.device_data(), reinterpret_cast<const void *>(d_keys.Current()), sizeof(float) * num_items,
                       cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(values.device_data(), reinterpret_cast<const void *>(d_values.Current()),
                          sizeof(int) * num_items,
                          cudaMemcpyDeviceToDevice));
};

#endif //THUNDERGBM_CUB_UTIL_H
