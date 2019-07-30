//
// Created by shijiashuai on 5/7/18.
//
#include <thundergbm/util/cub_wrapper.h>
#include <thundergbm/sparse_columns.h>

#include "thundergbm/sparse_columns.h"
#include "thundergbm/util/device_lambda.cuh"
#include "cusparse.h"
#include "thundergbm/util/multi_device.h"
#include "omp.h"

//FIXME remove this function
void correct_start(int *csc_col_ptr_2d_data, int first_col_start, int n_column_sub){
    device_loop(n_column_sub + 1, [=] __device__(int col_id) {
        csc_col_ptr_2d_data[col_id] = csc_col_ptr_2d_data[col_id] - first_col_start;
    });
};

void SparseColumns::csr2csc_gpu(const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &v_columns) {
    LOG(INFO) << "convert csr to csc using gpu...";
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

    //three arrays (on GPU/CPU) for csr representation
    this->column_offset = 0;
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
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    n_column = dataset.n_features_;
    n_row = dataset.n_instances();
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

    val.resize(0);
    row_ptr.resize(0);
    col_idx.resize(0);
    SyncMem::clear_cache();

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

    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO) << "Converting csr to csc using time: " << used_time.count() << " s";
}

void SparseColumns::csr2csc_cpu(const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &v_columns) {
    LOG(INFO) << "convert csr to csc using cpu...";
    this->column_offset = 0;
    //cpu transpose
    n_column = dataset.n_features();
    n_row = dataset.n_instances();
    nnz = dataset.csr_val.size();

    float_type *csc_val_ptr = new float_type[nnz];
    int *csc_row_ptr = new int[nnz];
    int *csc_col_ptr = new int[n_column + 1];

    LOG(INFO) << string_format("#non-zeros = %ld, density = %.2f%%", nnz,
                               (float) nnz / n_column / dataset.n_instances() * 100);
    for (int i = 0; i <= n_column; ++i) {
        csc_col_ptr[i] = 0;
    }

    #pragma omp parallel for // about 5s
    for (int i = 0; i < nnz; ++i) {
        int idx = dataset.csr_col_idx[i] + 1;
        #pragma omp atomic
        csc_col_ptr[idx] += 1;
    }

    for (int i = 1; i < n_column + 1; ++i){
        csc_col_ptr[i] += csc_col_ptr[i - 1];
    }

    // TODO to parallelize here
    for (int row = 0; row < dataset.n_instances(); ++row) {
        for (int j = dataset.csr_row_ptr[row]; j < dataset.csr_row_ptr[row + 1]; ++j) {
            int col = dataset.csr_col_idx[j]; // csr col
            int dest = csc_col_ptr[col]; // destination index in csc array
            csc_val_ptr[dest] = dataset.csr_val[j];
            csc_row_ptr[dest] = row;
            csc_col_ptr[col] += 1; //increment sscolumn start position
        }
    }

    //recover column start position
    for (int i = 0, last = 0; i < n_column; ++i) {
        int next_last = csc_col_ptr[i];
        csc_col_ptr[i] = last;
        last = next_last;
    }

    // split data to multiple device
    int n_device = v_columns.size();
    int ave_n_columns = n_column / n_device;
    DO_ON_MULTI_DEVICES(n_device, [&](int device_id){
        SparseColumns &columns = *v_columns[device_id];
        int first_col_id = device_id * ave_n_columns;
        int n_column_sub = (device_id < n_device - 1) ? ave_n_columns : n_column - first_col_id;
        int first_col_start = csc_col_ptr[first_col_id];
        int nnz_sub = (device_id < n_device - 1) ?
                      (csc_col_ptr[(device_id + 1) * ave_n_columns] - first_col_start) : (nnz - first_col_start);

        columns.column_offset = first_col_id + this->column_offset;
        columns.nnz = nnz_sub;
        columns.n_column = n_column_sub;
        columns.n_row = n_row;
        columns.csc_val.resize(nnz_sub);
        columns.csc_row_idx.resize(nnz_sub);
        columns.csc_col_ptr.resize(n_column_sub + 1);

        columns.csc_val.copy_from(csc_val_ptr + first_col_start, nnz_sub);
        columns.csc_row_idx.copy_from(csc_row_ptr + first_col_start, nnz_sub);
        columns.csc_col_ptr.copy_from(csc_col_ptr + first_col_id, n_column_sub + 1);

        int *csc_col_ptr_2d_data = columns.csc_col_ptr.host_data();
        correct_start(csc_col_ptr_2d_data, first_col_start, n_column_sub);
        seg_sort_by_key_cpu(columns.csc_val, columns.csc_row_idx, columns.csc_col_ptr);
    });

    delete[](csc_val_ptr);
    delete[](csc_row_ptr);
    delete[](csc_col_ptr);
}


void SparseColumns::csc_by_default(const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &v_columns) {
    const float_type *csc_val_ptr = dataset.csc_val.data();
    const int *csc_row_ptr = dataset.csc_row_idx.data();
    const int *csc_col_ptr = dataset.csc_col_ptr.data();
    n_column = dataset.n_features();
    n_row = dataset.n_instances();
    nnz = dataset.csc_val.size();

    // split data to multiple device
    int n_device = v_columns.size();
    int ave_n_columns = n_column / n_device;
    DO_ON_MULTI_DEVICES(n_device, [&](int device_id){
        SparseColumns &columns = *v_columns[device_id];
        int first_col_id = device_id * ave_n_columns;
        int n_column_sub = (device_id < n_device - 1) ? ave_n_columns : n_column - first_col_id;
        int first_col_start = csc_col_ptr[first_col_id];
        int nnz_sub = (device_id < n_device - 1) ?
                      (csc_col_ptr[(device_id + 1) * ave_n_columns] - first_col_start) : (nnz - first_col_start);

        columns.column_offset = first_col_id + this->column_offset;
        columns.nnz = nnz_sub;

        columns.n_column = n_column_sub;
        columns.n_row = n_row;
        columns.csc_val.resize(nnz_sub);
        columns.csc_row_idx.resize(nnz_sub);
        columns.csc_col_ptr.resize(n_column_sub + 1);

        columns.csc_val.copy_from(csc_val_ptr + first_col_start, nnz_sub);
        columns.csc_row_idx.copy_from(csc_row_ptr + first_col_start, nnz_sub);
        columns.csc_col_ptr.copy_from(csc_col_ptr + first_col_id, n_column_sub + 1);

        int *csc_col_ptr_2d_data = columns.csc_col_ptr.host_data();
        correct_start(csc_col_ptr_2d_data, first_col_start, n_column_sub);
        cub_seg_sort_by_key(columns.csc_val, columns.csc_row_idx, columns.csc_col_ptr, false);
    });
}