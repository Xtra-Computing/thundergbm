//
// Created by shijiashuai on 5/7/18.
//

#ifndef THUNDERGBM_INS_STAT_H
#define THUNDERGBM_INS_STAT_H

#include "syncarray.h"
#include "thundergbm/objective/objective_function.h"

class InsStat {
public:

    ///gradient and hessian
    SyncArray<GHPair> gh_pair;

    ///backup for bagging
    SyncArray<GHPair> gh_pair_backup;
    ///node id
    SyncArray<int> nid;
    ///target value
    SyncArray<float_type> y;
    ///predict value
    SyncArray<float_type> y_predict;

    std::unique_ptr<ObjectiveFunction> obj;

    int n_instances;

    InsStat() = default;

    explicit InsStat(size_t n_instances) {
        resize(n_instances);
    }

    void resize(size_t n_instances);

    void update_gradient();

    void reset_nid();

    void do_bagging();
};

#endif //THUNDERGBM_INS_STAT_H
