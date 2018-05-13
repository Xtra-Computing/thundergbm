//
// Created by qinbin on 2018/5/9.
//

#ifndef THUNDERGBM_HIST_CUT_H
#define THUNDERGBM_HIST_CUT_H
#include "thundergbm/thundergbm.h"
#include "thundergbm/dataset.h"
#include "thundergbm/tree.h"

class HistCut{
public:
//split_point[i] stores the split points of feature i
    std::vector<std::vector<float_type>> split_points;
    HistCut() {split_points.clear();};
    void get_split_points(SparseColumns& columns, InsStat& stats, int max_num_bins, int n_instances, int n_features);
};
#endif //THUNDERGBM_HIST_CUT_H
