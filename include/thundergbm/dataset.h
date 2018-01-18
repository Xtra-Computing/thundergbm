//
// Created by jiashuai on 18-1-17.
//

#ifndef THUNDERGBM_DATASET_H
#define THUNDERGBM_DATASET_H

#include "thundergbm.h"
#include "syncarray.h"

class DataSet {
public:
    ///one feature value and corresponding index
    struct node {
        node(int index, float_type value) : index(index), value(value) {}

        int index;
        float_type value;
    };

    ///two-dimension node vector
    typedef vector<vector<DataSet::node>> node2d;

    ///load dataset from file
    void load_from_file(string file_name);

    const node2d &instances() const;

    size_t n_features() const;

    size_t n_instances() const;

    const vector<float_type> &y() const;

private:
    ///labels of instances
    vector<float_type> y_;
    node2d instances_;
    size_t n_features_;
};

class SparseColumns {
public:
    SyncArray<float_type> csc_val;
    SyncArray<int> csc_row_ind;
    SyncArray<int> csc_col_ptr;
    size_t n_column;
    int nnz;

    SparseColumns(const DataSet &);

    ~SparseColumns();
};

#endif //THUNDERGBM_DATASET_H
