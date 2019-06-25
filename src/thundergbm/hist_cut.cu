//
// Created by qinbin on 2018/5/9.
//

#include "thundergbm/hist_cut.h"
#include "thundergbm/quantile_sketch.h"
#include "thundergbm/syncarray.h"
#include <sstream>
#include <omp.h>
#include <thundergbm/hist_cut.h>
#include <thundergbm/util/cub_wrapper.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <climits>
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/unique.h"


void HistCut::get_cut_points2(SparseColumns &columns, int max_num_bins, int n_instances) {
    LOG(INFO) << "Getting cut points... ";
    int n_column = columns.n_column;
    SyncArray<float> unique_vals(n_column * n_instances);
    SyncArray<int> tmp_row_ptr(n_column + 1);

    SyncArray<int> tmp_params(2);  // [0] --> num_cut_points, [1] --> max_num_bins
    int h_tmp_params[2] = {0, max_num_bins};
    tmp_params.copy_from(h_tmp_params, 2);

    // get the original csc data
    auto csc_val_data = columns.csc_val.device_data();
    auto csc_col_ptr_data = columns.csc_col_ptr.device_data();
    auto unique_vals_data = unique_vals.device_data();
    auto tmp_row_ptr_data = tmp_row_ptr.device_data();
    auto tmp_params_data = tmp_params.device_data();

    // start to get cut points
    device_loop(n_column, [=] __device__(int fid){
        int col_start = csc_col_ptr_data[fid];
        int col_len = csc_col_ptr_data[fid+1] - col_start;
        auto val_data = csc_val_data + col_start;
        auto unique_start = unique_vals_data + fid*n_instances;  // notice here
        int unique_len = thrust::unique_copy(thrust::device, val_data, val_data + col_len, unique_start) - unique_start;
        int n_cp = (unique_len <= tmp_params_data[1]) ? unique_len : tmp_params_data[1];
        tmp_row_ptr_data[fid+1] = unique_len;
        atomicAdd(&tmp_params_data[0], n_cp);
    });

    // merge the cut points
    tmp_params_data = tmp_params.host_data();
    cut_points_val.resize(tmp_params_data[0]);
    cut_row_ptr.resize(n_column + 1);
    cut_fid.resize(tmp_params_data[0]);

    cut_row_ptr.copy_from(tmp_row_ptr);
    auto cut_row_ptr_data = cut_row_ptr.host_data();
    tmp_row_ptr_data = tmp_row_ptr.host_data();
    for(int i = 1; i < (n_column + 1); i++) {
        if(tmp_row_ptr_data[i] <= tmp_params_data[1])
            cut_row_ptr_data[i] += cut_row_ptr_data[i-1];
        else
            cut_row_ptr_data[i] = cut_row_ptr_data[i-1] + max_num_bins;
    }

    auto cut_point_val_data = cut_points_val.device_data();
    tmp_row_ptr_data = tmp_row_ptr.device_data();
    tmp_params_data = tmp_params.device_data();
    cut_row_ptr_data = cut_row_ptr.device_data();
    unique_vals_data = unique_vals.device_data();
    device_loop_2d(n_column, cut_row_ptr.device_data(), [=] __device__ (int fid, int i){
        // fid --> [0, n_column)  &&  i -->> [cut_row_ptr[fid], cut_row_ptr[fid+1])
        int unique_len = tmp_row_ptr_data[fid+1];
        int unique_idx =  i - cut_row_ptr_data[fid];
        int cp_idx = (unique_len <= tmp_params_data[1]) ? unique_idx : (unique_len / tmp_params_data[1] * unique_idx);
        cut_point_val_data[i] = unique_vals_data[fid*n_instances + cp_idx];
    });

    auto cut_fid_data = cut_fid.device_data();
    device_loop_2d(n_column, cut_row_ptr.device_data(), [=] __device__(int fid, int i) {
        cut_fid_data[i] = fid;
    });
}

template<typename T>
void syncarray_resize(SyncArray<T> &buf_array, int new_size) {
    CHECK_GT(buf_array.size(), new_size) << "The size of the target Syncarray must greater than the new size. ";
    SyncArray<T> tmp_array(new_size);
    tmp_array.copy_from(buf_array.device_data(), new_size);
    buf_array.resize(new_size);
    buf_array.copy_from(tmp_array);
}


void unique_by_flag(SyncArray<float> &target_arr, SyncArray<int> &flags, int n_columns) {
    using namespace thrust::placeholders;

    float max_elem = max_elements(target_arr);
    LOG(INFO) << "################>>>: " << max_elem;
    CHECK_LT(max_elem + n_columns*(max_elem + 1),INT_MAX) << "Max_values is too large to be transformed";

    // 1. transform data into unique ranges
    thrust::transform(thrust::device,
                      target_arr.device_data(),
                      target_arr.device_end(),
                      flags.device_data(),
                      target_arr.device_data(),
                      (_1 + _2 * (max_elem + 1)));
    // 2. sort the transformed data
    sort_array(target_arr, false);
    thrust::reverse(thrust::device, flags.device_data(), flags.device_end());
    // 3. eliminate duplicates
    auto new_end = thrust::unique_by_key(thrust::device, target_arr.device_data(), target_arr.device_end(),
                                         flags.device_data());
    int new_size = new_end.first - target_arr.device_data();
    syncarray_resize(target_arr, new_size);
    syncarray_resize(flags, new_size);
    // 4. transform data back
    thrust::transform(thrust::device, target_arr.device_data(),
                      target_arr.device_end(),
                      flags.device_data(),
                      target_arr.device_data(),
                      (_1 - _2 * (max_elem + 1)));
    cub_sort_by_key(flags, target_arr);
}

void HistCut::get_cut_points3(SparseColumns &columns, int max_num_bins, int n_instances) {
    std::chrono::high_resolution_clock timer;
    auto time_1 = timer.now();

    LOG(INFO) << "Getting cut points... ";
    int n_column = columns.n_column;

    cut_points_val.resize(columns.csc_val.size());
    cut_row_ptr.resize(columns.csc_col_ptr.size());
    cut_fid.resize(columns.csc_val.size());
    cut_points_val.copy_from(columns.csc_val);

    auto cut_fid_data = cut_fid.device_data();
    device_loop_2d(n_column, columns.csc_col_ptr.device_data(), [=] __device__(int fid, int i) {
        cut_fid_data[i] = fid;
    });
    unique_by_flag(cut_points_val, cut_fid, n_column);

    cut_row_ptr.resize(n_column + 1);
    auto cut_row_ptr_data = cut_row_ptr.device_data();
    device_loop(cut_fid.size(), [=] __device__(int fid) {
        atomicAdd(cut_row_ptr_data + cut_fid_data[fid] + 1, 1);
    });
    thrust::inclusive_scan(thrust::device, cut_row_ptr_data, cut_row_ptr_data + cut_row_ptr.size(), cut_row_ptr_data);

    SyncArray<int> select_index(cut_fid.size());
    auto select_index_data = select_index.device_data();
    device_loop_2d_with_maximum(n_column, cut_row_ptr_data, max_num_bins, [=] __device__(int fid, int i, int interval) {
        int feature_idx = i - cut_row_ptr_data[fid];
        if(interval == 0)
            select_index_data[i] = 1;
        else if(feature_idx < max_num_bins)
            select_index_data[cut_row_ptr_data[fid] + interval * feature_idx] = 1;
    });

    cub_select(cut_fid, select_index);
    cub_select(cut_points_val, select_index);

    cut_fid_data = cut_fid.device_data();
    cut_row_ptr.resize(n_column + 1);
    cut_row_ptr_data = cut_row_ptr.device_data();
    device_loop(cut_fid.size(), [=] __device__(int fid) {
        atomicAdd(cut_row_ptr_data + cut_fid_data[fid] + 1, 1);
    });
    thrust::inclusive_scan(thrust::device, cut_row_ptr_data, cut_row_ptr_data + cut_row_ptr.size(), cut_row_ptr_data);


    auto time_5 = timer.now();
    std::chrono::duration<double> total_time = time_5 - time_1;
    LOG(INFO) << "  ----------------->>> Last LOOP: " << total_time.count();

    LOG(INFO) << "--->>>>  cut points value: " << cut_points_val;
    LOG(INFO) << "--->>>> cut row ptr: " << cut_row_ptr;
    LOG(INFO) << "--->>>> cut fid: " << cut_fid;
    LOG(INFO) << "TOTAL CP:" << cut_fid.size();
    LOG(INFO) << "NNZ: " << columns.csc_val.size();
}