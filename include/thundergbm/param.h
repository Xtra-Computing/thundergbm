//
// Created by shijiashuai on 5/7/18.
//

#ifndef THUNDERGBM_PARAM_H
#define THUNDERGBM_PARAM_H

#include "thundergbm/util/common.h"

struct GBMParam {
    int depth;
    int n_trees;
    float_type min_child_weight;
    float_type lambda;
    float_type gamma;
    float_type rt_eps;
    float column_sampling_rate;
    std::string path;
    bool verbose;
    bool bagging;
    int n_parallel_trees;
    float learning_rate;

    //for histogram
    int max_num_bin;

    int n_device;
};
#endif //THUNDERGBM_PARAM_H
