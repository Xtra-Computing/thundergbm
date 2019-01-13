//
// Created by zeyi on 1/9/19.
//

#ifndef THUNDERGBM_TRAINER_H
#define THUNDERGBM_TRAINER_H

#include "thundergbm.h"
#include "param.h"
#include "tree.h"
#include "dataset.h"

class TreeTrainer{
public:
    float_type train(GBMParam &param);
    float_type train_exact(GBMParam &param);
    float_type train_hist(GBMParam &param);

    float_type compute_rmse(const InsStat &stats);

    void save_trees(GBMParam &param, vector<Tree> &trees);
private:
    DataSet dataSet;
};

#endif //THUNDERGBM_TRAINER_H
