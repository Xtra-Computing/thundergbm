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
    int n_column;
    int n_row;
    int column_offset;
    int nnz;

    void csr2csc_gpu(const DataSet &dataSet, vector<std::unique_ptr<SparseColumns>> &);
    void csr2csc_cpu(const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &);
    void csc_by_default(const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &v_columns);
    void to_multi_devices(vector<std::unique_ptr<SparseColumns>> &) const;

};
#endif //THUNDERGBM_SPARSE_COLUMNS_H
