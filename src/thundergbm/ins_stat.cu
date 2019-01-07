//
// Created by shijiashuai on 5/7/18.
//
#include "thundergbm/ins_stat.h"
//#include "thrust/reduce.h"
#include "thrust/random.h"
#include "thundergbm/util/device_lambda.cuh"

void InsStat::resize(size_t n_instances) {
    this->n_instances = n_instances;
    gh_pair.resize(n_instances);
    nid.resize(n_instances);
    y.resize(n_instances);
    y_predict.resize(n_instances);
}

void InsStat::update_gradient() {
    LOG(DEBUG) << y_predict;
    obj->get_gradient(y, y_predict, gh_pair);
}

void InsStat::reset_nid() {
    auto nid_data = nid.device_data();
    device_loop(n_instances, [=]__device__(int i) {
        nid_data[i] = 0;
    });
}

void InsStat::do_bagging() {
    LOG(TRACE) << "do bagging";
    using namespace thrust;
    SyncArray<int> idx(n_instances);
    auto idx_data = idx.device_data();
    int n_instances = this->n_instances;
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
    gh_pair.copy_from(gh_pair_backup);
    auto gh_data = gh_pair.device_data();
    //FIXME synchronize between shards
    device_loop(n_instances, [=]__device__(int i) {
        gh_data[i].g = gh_data[i].g * ins_count_data[i];
        gh_data[i].h = gh_data[i].h * ins_count_data[i];
    });
}

