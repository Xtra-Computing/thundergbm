//
// Created by shijiashuai on 5/7/18.
//

#ifndef THUNDERGBM_PARAM_H
#define THUNDERGBM_PARAM_H

#include "thundergbm.h"

struct GBMParam {
    int depth;
    int n_trees;
    float_type min_child_weight;
    float_type lambda;
    float_type gamma;
    float_type rt_eps;
    string path;
    bool do_exact = true;

    //for histogram
    int max_num_bin = 256;

    int n_device;
    int n_executor;
};
#endif //THUNDERGBM_PARAM_H
