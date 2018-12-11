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
    std::string path;
    bool verbose;

    //for histogram
    int max_num_bin = 255;

    int n_device;
};
#endif //THUNDERGBM_PARAM_H
