//
// Created by ss on 19-1-3.
//

#ifndef THUNDERGBM_MULTICLASS_OBJ_H
#define THUNDERGBM_MULTICLASS_OBJ_H

#include "objective_function.h"
#include "thundergbm/util/device_lambda.cuh"

class Softmax : public ObjectiveFunction {
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override;

    void predict_transform(SyncArray<float_type> &y) override;

    void configure(GBMParam param, const DataSet &dataset) override;

    string default_metric_name() override { return "macc"; }

    virtual ~Softmax() override = default;

protected:
    int num_class;
    SyncArray<float_type> label;
};


class SoftmaxProb : public Softmax {
public:
    void predict_transform(SyncArray<float_type> &y) override;

    ~SoftmaxProb() override = default;

};



#endif //THUNDERGBM_MULTICLASS_OBJ_H
