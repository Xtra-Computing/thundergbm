//
// Created by zeyi on 1/12/19.
//

#ifndef THUNDERGBM_PREDICTOR_H
#define THUNDERGBM_PREDICTOR_H

#include "thundergbm/tree.h"
#include <thundergbm/dataset.h>

class Predictor{
public:
    vector<float_type> predict(GBMParam &model_param, vector<vector<Tree>> &boosted_model, DataSet &dataSet);
};

#endif //THUNDERGBM_PREDICTOR_H
