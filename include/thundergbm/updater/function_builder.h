//
// Created by ss on 19-1-17.
//

#ifndef THUNDERGBM_FUNCTION_BUILDER_H
#define THUNDERGBM_FUNCTION_BUILDER_H

#include <thundergbm/tree.h>
#include "thundergbm/common.h"
#include "thundergbm/sparse_columns.h"

class FunctionBuilder {
public:
    virtual vector<Tree> build_approximate(const MSyncArray<GHPair> &gradients) = 0;

    virtual void init(const DataSet &dataset, const GBMParam &param) {};

    virtual const MSyncArray<float_type> &get_y_predict() = 0;

    static FunctionBuilder *create(std::string name);
};

#endif //THUNDERGBM_FUNCTION_BUILDER_H
