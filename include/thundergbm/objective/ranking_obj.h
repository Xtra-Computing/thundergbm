//
// Created by ss on 19-1-10.
//

#ifndef THUNDERGBM_RANKING_OBJ_H
#define THUNDERGBM_RANKING_OBJ_H

#include "objective_function.h"
#include "thundergbm/util/device_lambda.cuh"

/**
 *
 * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
 */
class LambdaRank : public ObjectiveFunction {
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override;

    void configure(GBMParam param, const DataSet &dataset) override;

    string default_metric() override;

    ~LambdaRank() override = default;

private:
    vector<int> gptr;//group start position
    vector<float_type> idcg;
    int n_group;

    float_type sigma;
};



#endif //THUNDERGBM_RANKING_OBJ_H
