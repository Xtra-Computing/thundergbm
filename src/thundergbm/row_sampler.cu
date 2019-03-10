//
// Created by shijiashuai on 2019-02-15.
//
#include <thundergbm/row_sampler.h>
#include "thundergbm/util/multi_device.h"
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/random.h"

void RowSampler::do_bagging(MSyncArray<GHPair> &gradients) {
    LOG(TRACE) << "do bagging";
    using namespace thrust;
    int n_instances = gradients.front().size();
    SyncArray<int> idx(n_instances);
    auto idx_data = idx.device_data();
    int seed = std::rand();//TODO add a global random generator class
    device_loop(n_instances, [=]__device__(int i) {
        default_random_engine rng(seed);
        uniform_int_distribution<int> uniform_dist(0, n_instances - 1);
        rng.discard(i);
        idx_data[i] = uniform_dist(rng);
    });
    SyncArray<int> ins_count(n_instances);
    auto ins_count_data = ins_count.device_data();
    device_loop(n_instances, [=]__device__(int i) {
        int ins_id = idx_data[i];
        atomicAdd(ins_count_data + ins_id, 1);
    });
    DO_ON_MULTI_DEVICES(gradients.size(), [&](int device_id){
        auto gh_data = gradients[device_id].device_data();
        device_loop(n_instances, [=]__device__(int i) {
            gh_data[i].g = gh_data[i].g * ins_count_data[i];
            gh_data[i].h = gh_data[i].h * ins_count_data[i];
        });
    });
}

