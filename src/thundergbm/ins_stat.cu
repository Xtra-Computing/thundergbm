//
// Created by shijiashuai on 5/7/18.
//
#include "thundergbm/ins_stat.h"
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/reduce.h"

void InsStat::resize(size_t n_instances) {
    this->n_instances = n_instances;
    gh_pair.resize(n_instances);
    nid.resize(n_instances);
    y.resize(n_instances);
    y_predict.resize(n_instances);
}

void InsStat::updateGH() {
    auto gh_pair_data = gh_pair.device_data();
    auto nid_data = nid.device_data();
    auto stats_y_data = y.device_data();
    auto stats_yp_data = y_predict.device_data();
    LOG(DEBUG) << y_predict;
    LOG(TRACE) << "initializing instance statistics";
    device_loop(n_instances, [=]__device__(int i){
        nid_data[i] = 0;
        //TODO support other objective function
        gh_pair_data[i].g = stats_yp_data[i] - stats_y_data[i];
        gh_pair_data[i].h = 1;
    });
    sum_gh = thrust::reduce(thrust::cuda::par, gh_pair.device_data(), gh_pair.device_end());
}
