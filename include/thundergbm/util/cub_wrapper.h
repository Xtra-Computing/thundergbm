//
// Created by ss on 18-5-13.
//

#ifndef THUNDERGBM_CUB_UTIL_H
#define THUNDERGBM_CUB_UTIL_H

#include <thundergbm/syncarray.h>
#include "cub/cub.cuh"
#include "thrust/sort.h"

template<typename T1, typename T2>
void cub_sort_by_key(SyncArray<T1> &keys, SyncArray<T2> &values, int size = -1, bool ascending = true,
                     void *temp = nullptr) {
    CHECK_EQ(values.size(), values.size()) << "keys and values must have equal size";
    using namespace cub;
    size_t num_items;
    if (-1 == size)
        num_items = keys.size();
    else
        num_items = size;
    SyncArray<char> temp_storage;
    DoubleBuffer<T1> d_keys;
    DoubleBuffer<T2> d_values;
    if (!temp) {
        SyncArray<T1> keys2(num_items);
        SyncArray<T2> values2(num_items);

        d_keys = DoubleBuffer<T1>(keys.device_data(), keys2.device_data());
        d_values = DoubleBuffer<T2>(values.device_data(), values2.device_data());
    } else {
        d_keys = DoubleBuffer<T1>(keys.device_data(), (T1 *) temp);
        d_values = DoubleBuffer<T2>(values.device_data(), (T2 *) ((T1 *) temp + num_items));
    }

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
            cudaMemcpy(keys.device_data(), reinterpret_cast<const void *>(d_keys.Current()), sizeof(T1) * num_items,
                       cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(values.device_data(), reinterpret_cast<const void *>(d_values.Current()),
                          sizeof(T2) * num_items,
                          cudaMemcpyDeviceToDevice));
}

template<typename T1, typename T2>
void cub_seg_sort_by_key(SyncArray<T1> &keys, SyncArray<T2> &values, const SyncArray<int> &ptr, bool ascending = true) {
    CHECK_EQ(keys.size(), values.size()) << "keys and values must have equal size";
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
            cudaMemcpy(keys.device_data(), reinterpret_cast<const void *>(d_keys.Current()), sizeof(T1) * num_items,
                       cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(values.device_data(), reinterpret_cast<const void *>(d_values.Current()),
                          sizeof(T2) * num_items,
                          cudaMemcpyDeviceToDevice));
};

template<typename T>
void sort_array(SyncArray<T> &in_arr, bool ascending = true) {
    CHECK_GT(in_arr.size(), 0) << "The size of target array must greater than 0. ";

    int num_items = in_arr.size();
    SyncArray<T> out_arr(num_items);
    cub::DoubleBuffer<T> d_keys(in_arr.device_data(), out_arr.device_data());

    size_t temp_storage_bytes = 0;
    SyncArray<char> temp_storage;
    if (ascending)
        cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes, d_keys, num_items);
    else
        cub::DeviceRadixSort::SortKeysDescending(NULL, temp_storage_bytes, d_keys, num_items);
    temp_storage.resize(temp_storage_bytes);
    if (ascending)
        cub::DeviceRadixSort::SortKeys(temp_storage.device_data(), temp_storage_bytes, d_keys, num_items);
    else
        cub::DeviceRadixSort::SortKeysDescending(temp_storage.device_data(), temp_storage_bytes, d_keys, num_items);
    CUDA_CHECK(
            cudaMemcpy(in_arr.device_data(), reinterpret_cast<const void *>(d_keys.Current()), sizeof(T) * num_items,
                       cudaMemcpyDeviceToDevice));
}

template<typename T>
T max_elements(SyncArray<T> &target_arr) {
    CHECK_GT(target_arr.size(), 0) << "The size of target array must greater than 0. ";

    int num_items = target_arr.size();
    size_t temp_storage_bytes = 0;
    SyncArray<char> temp_storage;
    SyncArray<T> max_result(1);
    cub::DeviceReduce::Max(NULL, temp_storage_bytes, target_arr.device_data(), max_result.device_data(), num_items);
    temp_storage.resize(temp_storage_bytes);
    cub::DeviceReduce::Max(temp_storage.device_data(), temp_storage_bytes, target_arr.device_data(), max_result.device_data(), num_items);

    return *max_result.host_data();
}

template<typename T>
void cub_select(SyncArray<T> &in_arr, const SyncArray<int> &flags) {
    CHECK_EQ(in_arr.size(), flags.size()) << "Size of in_array must equals to flags array. ";

    int num_items = in_arr.size();
    SyncArray<T> out_arr(num_items);
    SyncArray<int> num_selected(1);
    SyncArray<char> temp_storage;
    size_t temp_storage_bytes = 0;

    cub::DeviceSelect::Flagged(NULL, temp_storage_bytes, in_arr.device_data(), flags.device_data(),
                               out_arr.device_data(), num_selected.device_data(), num_items);
    temp_storage.resize(temp_storage_bytes);
    cub::DeviceSelect::Flagged(temp_storage.device_data(), temp_storage_bytes, in_arr.device_data(), flags.device_data(),
                               out_arr.device_data(), num_selected.device_data(), num_items);

    int new_size = *num_selected.host_data();
    in_arr.resize(new_size);
    in_arr.copy_from(out_arr.device_data(), new_size);
}

template<typename T1, typename T2>
void seg_sort_by_key_cpu(SyncArray<T1> &keys, SyncArray<T2> &values, const SyncArray<int> &ptr) {
    auto keys_data = keys.device_data();
    auto values_data = values.device_data();
    auto offset_data = ptr.host_data();
    LOG(INFO) << ptr;
#
    for(int i = 0; i < ptr.size() - 2; i++)
    {
        int seg_len = offset_data[i + 1] - offset_data[i];
        auto key_start = keys_data + offset_data[i];
        auto key_end = key_start + seg_len;
        auto value_start = values_data + offset_data[i];
        thrust::sort_by_key(thrust::device, key_start, key_end, value_start, thrust::greater<T1>());
    }
}

#endif //THUNDERGBM_CUB_UTIL_H
