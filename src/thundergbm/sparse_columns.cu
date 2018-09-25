//
// Created by shijiashuai on 5/7/18.
//
#include <thundergbm/util/cub_wrapper.h>
#include "thundergbm/sparse_columns.h"
#include "thundergbm/util/device_lambda.cuh"

void SparseColumns::from_dataset(const DataSet &dataset) {
    LOG(TRACE) << "constructing sparse columns";
    n_column = dataset.n_features();
    vector<float_type> csc_val_vec;
    vector<int> csc_row_ind_vec;
    vector<int> csc_col_ptr_vec;
    csc_col_ptr_vec.push_back(0);
    for (int i = 0; i < n_column; i++) {
        csc_val_vec.insert(csc_val_vec.end(), dataset.features[i].begin(), dataset.features[i].end());
        csc_row_ind_vec.insert(csc_row_ind_vec.end(), dataset.line_num[i].begin(), dataset.line_num[i].end());
        csc_col_ptr_vec.push_back(csc_col_ptr_vec.back() + dataset.features[i].size());
    }
    nnz = csc_val_vec.size();
    csc_val.resize(csc_val_vec.size());
    memcpy(csc_val.host_data(), csc_val_vec.data(), sizeof(float_type) * csc_val_vec.size());
    csc_row_ind.resize(csc_row_ind_vec.size());
    memcpy(csc_row_ind.host_data(), csc_row_ind_vec.data(), sizeof(int) * csc_row_ind_vec.size());
    csc_col_ptr.resize(csc_col_ptr_vec.size());
    memcpy(csc_col_ptr.host_data(), csc_col_ptr_vec.data(), sizeof(int) * csc_col_ptr_vec.size());
    cudaDeviceSynchronize();// ?
}


void SparseColumns::to_multi_devices(vector<std::shared_ptr<SparseColumns>> &v_columns) const {
    //devide data into multiple devices
    int n_device = v_columns.size();
    int ave_n_columns = n_column / n_device;
    DO_ON_MULTI_DEVICES(n_device, [&](int device_id) {
        SparseColumns &columns = *v_columns[device_id];
        const int *csc_col_ptr_data = csc_col_ptr.host_data();
        int first_col_id = device_id * ave_n_columns;
        int n_column_sub = (device_id < n_device - 1) ? ave_n_columns : n_column - first_col_id;
        int first_col_start = csc_col_ptr_data[first_col_id];
        int nnz_sub = (device_id < n_device - 1) ?
                      (csc_col_ptr_data[(device_id + 1) * ave_n_columns] - first_col_start) : (nnz -
                                                                                               first_col_start);
        columns.column_offset = first_col_id + this->column_offset;
        columns.nnz = nnz_sub;
        columns.n_column = n_column_sub;
        columns.csc_val.resize(nnz_sub);
        columns.csc_row_ind.resize(nnz_sub);
        columns.csc_col_ptr.resize(n_column_sub + 1);

        columns.csc_val.copy_from(csc_val.host_data() + first_col_start, nnz_sub);
        columns.csc_row_ind.copy_from(csc_row_ind.host_data() + first_col_start, nnz_sub);
        columns.csc_col_ptr.copy_from(csc_col_ptr.host_data() + first_col_id, n_column_sub + 1);

        int *csc_col_ptr_2d_data = columns.csc_col_ptr.device_data();

        //correct segment start positions
        device_loop(n_column_sub + 1, [=] __device__(int col_id) {
            csc_col_ptr_2d_data[col_id] = csc_col_ptr_2d_data[col_id] - first_col_start;
        });
        LOG(TRACE) << "sorting feature values (multi-device)";
        cub_seg_sort_by_key(columns.csc_val, columns.csc_row_ind, columns.csc_col_ptr, false);
    });
    LOG(TRACE) << "sorting finished";
}

void SparseColumns::get_shards(int rank, int n, SparseColumns &result) const {
    //devide data into multiple devices
    int ave_n_columns = n_column / n;
    SparseColumns &columns = result;
    const int *csc_col_ptr_data = csc_col_ptr.host_data();
    int first_col_id = rank * ave_n_columns;
    int n_column_sub = (rank < n - 1) ? ave_n_columns : n_column - first_col_id;
    int first_col_start = csc_col_ptr_data[first_col_id];
    int nnz_sub = (rank < n - 1) ?
                  (csc_col_ptr_data[(rank + 1) * ave_n_columns] - first_col_start) : (nnz -
                                                                                      first_col_start);
    columns.column_offset = first_col_id;
    columns.nnz = nnz_sub;
    columns.n_column = n_column_sub;
    columns.csc_val.resize(nnz_sub);
    columns.csc_row_ind.resize(nnz_sub);
    columns.csc_col_ptr.resize(n_column_sub + 1);

    columns.csc_val.copy_from(csc_val.host_data() + first_col_start, nnz_sub);
    columns.csc_row_ind.copy_from(csc_row_ind.host_data() + first_col_start, nnz_sub);
    columns.csc_col_ptr.copy_from(csc_col_ptr.host_data() + first_col_id, n_column_sub + 1);

    int *csc_col_ptr_2d_data = columns.csc_col_ptr.device_data();

    //correct segment start positions
    device_loop(n_column_sub + 1, [=] __device__(int col_id) {
        csc_col_ptr_2d_data[col_id] = csc_col_ptr_2d_data[col_id] - first_col_start;
    });
    LOG(TRACE) << "sorting feature values (multi-device)";
    cub_seg_sort_by_key(columns.csc_val, columns.csc_row_ind, columns.csc_col_ptr, false);
}
