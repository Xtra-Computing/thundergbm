//
// Created by shijiashuai on 2019-02-15.
//

#ifndef THUNDERGBM_ROW_SAMPLER_H
#define THUNDERGBM_ROW_SAMPLER_H

#include "thundergbm/common.h"
#include "syncarray.h"

class RowSampler {
public:
    void do_bagging(MSyncArray<GHPair> &gradients);
};
#endif //THUNDERGBM_ROW_SAMPLER_H
