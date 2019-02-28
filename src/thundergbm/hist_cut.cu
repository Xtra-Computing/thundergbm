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
    int n_column = columns.n_column;
    auto csc_val_data = columns.csc_val.host_data();
    auto csc_col_ptr_data = columns.csc_col_ptr.host_data();
    cut_points.clear();
    row_ptr.clear();
    row_ptr.resize(1, 0);

    //TODO do this on GPU
    for (int fid = 0; fid < n_column; ++fid) {
        int col_start = csc_col_ptr_data[fid];
        int col_len = csc_col_ptr_data[fid + 1] - col_start;
        auto val_data = csc_val_data + col_start;
        vector<float_type> unique_val(col_len);

        int unique_len = thrust::unique_copy(thrust::host, val_data, val_data + col_len, unique_val.data()) - unique_val.data();
        if (unique_len <= max_num_bins) {
            row_ptr.push_back(unique_len + row_ptr.back());
            for (int i = 0; i < unique_len; ++i) {
                cut_points.push_back(unique_val[i]);
            }
        } else {
            row_ptr.push_back(max_num_bins + row_ptr.back());
            for (int i = 0; i < max_num_bins; ++i) {
                cut_points.push_back(unique_val[unique_len / max_num_bins * i]);
            }
        }
    }

    cut_points_val.resize(cut_points.size());
    cut_points_val.copy_from(cut_points.data(), cut_points.size());
    cut_row_ptr.resize(row_ptr.size());
    cut_row_ptr.copy_from(row_ptr.data(), row_ptr.size());
    cut_fid.resize(cut_points.size());
    auto cut_fid_data = cut_fid.device_data();
    device_loop_2d(n_column, cut_row_ptr.device_data(), [=] __device__(int fid, int i) {
        cut_fid_data[i] = fid;
    });
}
