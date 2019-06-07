//
// Created by qinbin on 2018/5/9.
//

#include "thundergbm/hist_cut.h"
#include "thundergbm/quantile_sketch.h"
#include "thundergbm/syncarray.h"
#include <sstream>
#include <omp.h>
#include <thundergbm/hist_cut.h>

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
