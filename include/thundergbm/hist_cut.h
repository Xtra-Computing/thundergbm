//
// Created by qinbin on 2018/5/9.
//

#ifndef THUNDERGBM_HIST_CUT_H
#define THUNDERGBM_HIST_CUT_H

#include "common.h"
#include "sparse_columns.h"
#include "thundergbm/dataset.h"
#include "thundergbm/tree.h"
#include "ins_stat.h"

class HistCut {
public:
//split_point[i] stores the split points of feature i
    //std::vector<std::vector<float_type>> split_points;
    vector<float_type> cut_points;
    vector<int> row_ptr;
    //for gpu
    SyncArray<float_type> cut_points_val;
    SyncArray<int> cut_row_ptr;
    SyncArray<int> cut_fid;

    HistCut() = default;

    HistCut(const HistCut &cut) {
        cut_points = cut.cut_points;
        row_ptr = cut.row_ptr;
        cut_points_val.copy_from(cut.cut_points_val);
        cut_row_ptr.copy_from(cut.cut_row_ptr);
    }

    void get_cut_points2(SparseColumns &columns, int max_num_bins, int n_instances);
    void get_cut_points3(SparseColumns &columns, int max_num_bins, int n_instances);
};

#endif //THUNDERGBM_HIST_CUT_H
