//
// Created by ss on 19-1-1.
//
#include <thundergbm/objective/objective_function.h>
#include "thundergbm/objective/regression_obj.h"

ObjectiveFunction *ObjectiveFunction::create(string name) {
    if (name == "reg:linear") return new RegressionObj<SquareLoss>;
    if (name == "reg:logistic") return new RegressionObj<LogisticLoss>;
    return nullptr;
}
