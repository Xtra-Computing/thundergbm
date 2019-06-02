//
// Created by shijiashuai on 5/7/18.
//
#include <thundergbm/util/cub_wrapper.h>
#include <thundergbm/sparse_columns.h>

#include "thundergbm/sparse_columns.h"
#include "thundergbm/util/device_lambda.cuh"
#include "cusparse.h"
#include "thundergbm/util/multi_device.h"

void SparseColumns::from_dataset(const DataSet &dataset) {
    LOG(INFO) << "copy csr matrix to GPU";
    //three arrays (on GPU/CPU) for csr representation
    SyncArray<float_type> val;
    SyncArray<int> col_idx;
    SyncArray<int> row_ptr;
    val.resize(dataset.csr_val.size());
    col_idx.resize(dataset.csr_col_idx.size());
    row_ptr.resize(dataset.csr_row_ptr.size());

    //copy data to the three arrays
    val.copy_from(dataset.csr_val.data(), val.size());
    col_idx.copy_from(dataset.csr_col_idx.data(), col_idx.size());
    row_ptr.copy_from(dataset.csr_row_ptr.data(), row_ptr.size());
    LOG(INFO) << "converting csr matrix to csc matrix";
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    n_column = dataset.n_features_;
    nnz = dataset.csr_val.size();
    csc_val.resize(nnz);
    csc_row_idx.resize(nnz);
    csc_col_ptr.resize(n_column + 1);

    cusparseScsr2csc(handle, dataset.n_instances(), n_column, nnz, val.device_data(), row_ptr.device_data(),
                     col_idx.device_data(), csc_val.device_data(), csc_row_idx.device_data(), csc_col_ptr.device_data(),
                     CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);
}


//FIXME remove this function
void correct_start(int *csc_col_ptr_2d_data, int first_col_start, int n_column_sub){
    device_loop(n_column_sub + 1, [=] __device__(int col_id) {
        csc_col_ptr_2d_data[col_id] = csc_col_ptr_2d_data[col_id] - first_col_start;
    });
};
void SparseColumns::to_multi_devices(vector<std::unique_ptr<SparseColumns>> &v_columns) const {
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
        columns.n_row = n_row;
        columns.csc_val.resize(nnz_sub);
        columns.csc_row_idx.resize(nnz_sub);
        columns.csc_col_ptr.resize(n_column_sub + 1);

        columns.csc_val.copy_from(csc_val.host_data() + first_col_start, nnz_sub);
        columns.csc_row_idx.copy_from(csc_row_idx.host_data() + first_col_start, nnz_sub);
        columns.csc_col_ptr.copy_from(csc_col_ptr.host_data() + first_col_id, n_column_sub + 1);

        int *csc_col_ptr_2d_data = columns.csc_col_ptr.device_data();
        correct_start(csc_col_ptr_2d_data, first_col_start, n_column_sub);
        //correct segment start positions
        LOG(TRACE) << "sorting feature values (multi-device)";
        cub_seg_sort_by_key(columns.csc_val, columns.csc_row_idx, columns.csc_col_ptr, false);
    });
    LOG(TRACE) << "sorting finished";
}

