//
// Created by qinbin on 2018/5/9.
//

#ifndef THUNDERGBM_HIST_CUT_H
#define THUNDERGBM_HIST_CUT_H

#include "thundergbm/thundergbm.h"
#include "thundergbm/dataset.h"
#include "thundergbm/tree.h"
#include "sparse_columns.h"

class HistCut {
public:
//split_point[i] stores the split points of feature i
    //std::vector<std::vector<float_type>> split_points;
    vector<float_type> cut_points;
    vector<int> row_ptr;
    //for gpu
    SyncArray<float_type> cut_points_val;
    SyncArray<int> cut_row_ptr;

    HistCut() = default;

    HistCut(const HistCut &cut) {
        cut_points = cut.cut_points;
        row_ptr = cut.row_ptr;
        cut_points_val.copy_from(cut.cut_points_val);
        cut_row_ptr.copy_from(cut.cut_row_ptr);
    }

    void get_cut_points(SparseColumns &columns, InsStat &stats, int max_num_bins, int n_instances, int n_features);
};

//store the g/h of the bins of one feature
class BinStat {
public:
    SyncArray<GHPair> gh_pair;
    //feature id
    int fid;
    //number of bins
    int numBin;

    BinStat() = default;

    //feature: the pointer to features that need to build hist
    //insId: the pointer to instance id of features
    void Init(HistCut &cut, InsStat &stats, int pid, float_type *f_val, int n_f_val, int *iid);
    //void Init(vector<float_type>& cut_points, InsStat& stats,SparseColumns& columns, int fid);
};

#endif //THUNDERGBM_HIST_CUT_H
