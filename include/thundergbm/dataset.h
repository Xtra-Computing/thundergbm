//
// Created by jiashuai on 18-1-17.
//

#ifndef THUNDERGBM_DATASET_H
#define THUNDERGBM_DATASET_H

#include "thundergbm.h"
#include "syncarray.h"

class DataSet {
public:
    ///load dataset from file
    void load_from_sparse(int row_size, float* val, int* row_ptr, int* col_ptr, float* label);
    void load_from_file(string file_name, GBMParam param);
    void load_group_file(string file_name);

    size_t n_features() const;

    size_t n_instances() const;

    vector<float_type> csr_val;
    vector<int> csr_row_ptr;
    vector<int> csr_col_idx;
    vector<float_type> y;
    size_t n_features_;
    vector<int> group;
};

#endif //THUNDERGBM_DATASET_H
