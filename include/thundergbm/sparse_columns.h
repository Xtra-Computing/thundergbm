//
// Created by shijiashuai on 5/7/18.
//

#ifndef THUNDERGBM_SPARSE_COLUMNS_H
#define THUNDERGBM_SPARSE_COLUMNS_H

#include "syncarray.h"
#include "dataset.h"

class SparseColumns {//one feature corresponding to one column
public:
    SyncArray<float_type> csc_val;
    SyncArray<int> csc_row_idx;
    SyncArray<int> csc_col_ptr;
    
    //original order without sort
    SyncArray<float_type> csc_val_origin;
    SyncArray<int> csc_row_idx_origin;
    SyncArray<int> csc_col_ptr_origin;
    
    //csr data
    SyncArray<float_type> csr_val;
    SyncArray<int> csr_row_ptr;
    SyncArray<int> csr_col_idx;

    int max_trick_depth = -1;
    int max_trick_nodes = -1;

    int n_column;
    int n_row;
    int column_offset;
    size_t nnz;

    void csr2csc_gpu(const DataSet &dataSet, vector<std::unique_ptr<SparseColumns>> &);
    
    //function for data transfer to gpu , this function is only for single GPU device
    void to_gpu(const DataSet &dataSet, vector<std::unique_ptr<SparseColumns>> &);

    void csr2csc_cpu(const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &);
    void csc_by_default(const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &v_columns);
    void to_multi_devices(vector<std::unique_ptr<SparseColumns>> &) const;

};
#endif //THUNDERGBM_SPARSE_COLUMNS_H
