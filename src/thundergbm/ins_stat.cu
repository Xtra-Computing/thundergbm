//
// Created by shijiashuai on 5/7/18.
//
#include "thundergbm/ins_stat.h"

void InsStat::resize(size_t n_instances) {
    this->n_instances = n_instances;
    gh_pair.resize(n_instances);
    nid.resize(n_instances);
    y.resize(n_instances);
    y_predict.resize(n_instances);
}

void InsStat::updateGH() {
    sum_gh = GHPair(0, 0);
    GHPair *gh_pair_data = gh_pair.host_data();
    unsigned char *nid_data = nid.host_data();
    float_type *stats_y_data = y.host_data();
    float_type *stats_yp_data = y_predict.host_data();
    LOG(DEBUG) << y_predict;
    LOG(TRACE) << "initializing instance statistics";
    //TODO parallel?
    for (int i = 0; i < n_instances; ++i) {
        nid_data[i] = 0;
        //TODO support other objective function
        gh_pair_data[i].g = stats_yp_data[i] - stats_y_data[i];
        gh_pair_data[i].h = 1;
        sum_gh = sum_gh + gh_pair_data[i];
    }
}
