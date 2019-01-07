//
// Created by ss on 19-1-1.
//

#ifndef THUNDERGBM_OBJECTIVE_FUNCTION_H
#define THUNDERGBM_OBJECTIVE_FUNCTION_H

#include <thundergbm/syncarray.h>
#include "thundergbm/thundergbm.h"

class ObjectiveFunction {
public:
    //todo different target type
    virtual void
    get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p, SyncArray<GHPair> &gh_pair) = 0;
    virtual void
    predict_transform(SyncArray<float_type> &y) = 0;
    virtual void configure(GBMParam param) = 0;

    static ObjectiveFunction* create(string name);
    virtual ~ObjectiveFunction() = default;
};

#endif //THUNDERGBM_OBJECTIVE_FUNCTION_H
