//
// Created by zeyi on 1/9/19.
//

#ifndef THUNDERGBM_TRAINER_H
#define THUNDERGBM_TRAINER_H

#include "common.h"
#include "tree.h"
#include "dataset.h"

class TreeTrainer{
public:
    void train(GBMParam &param);
//    float_type train(GBMParam &param);
//    float_type train_exact(GBMParam &param);
//    float_type train_hist(GBMParam &param);

    void save_trees(GBMParam &param, vector<Tree> &trees);
private:
    DataSet dataSet;
};

#endif //THUNDERGBM_TRAINER_H
