//
// Created by ss on 19-1-1.
//
#include <thundergbm/objective/objective_function.h>
#include "thundergbm/objective/regression_obj.h"
#include "thundergbm/objective/multiclass_obj.h"
#include "thundergbm/objective/ranking_obj.h"

ObjectiveFunction *ObjectiveFunction::create(string name) {
    if (name == "reg:linear") return new RegressionObj<SquareLoss>;
    if (name == "reg:logistic") return new RegressionObj<LogisticLoss>;
    if (name == "binary:logistic") return new LogClsObj<LogisticLoss>;
    if (name == "multi:softprob") return new SoftmaxProb;
    if (name == "multi:softmax") return new Softmax;
    if (name == "rank:pairwise") return new LambdaRank;
    if (name == "rank:ndcg") return new LambdaRankNDCG;
    LOG(FATAL) << "undefined objective " << name;
    return nullptr;
}

bool ObjectiveFunction::need_load_group_file(string name) {
    return name == "rank:ndcg" || name == "rank:pairwise";
}

bool ObjectiveFunction::need_group_label(string name) {
    return name == "multi:softprob" || name == "multi:softmax" || name == "binary:logistic";
}
