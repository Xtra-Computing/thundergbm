//
// Created by zeyi on 1/12/19.
//

#ifndef THUNDERGBM_PREDICTOR_H
#define THUNDERGBM_PREDICTOR_H

#include "thundergbm/tree.h"
#include <thundergbm/dataset.h>

class Predictor{
public:
    void predict(vector<Tree> &trees, DataSet &dataSet);
private:
    int get_next_child(const Tree::TreeNode &node, float_type feaValue);
};

#endif //THUNDERGBM_PREDICTOR_H
