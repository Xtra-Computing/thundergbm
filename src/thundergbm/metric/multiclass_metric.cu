//
// Created by ss on 19-1-15.
//
#include "thundergbm/metric/multiclass_metric.h"
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/reduce.h"


float_type MulticlassAccuracy::get_score(const SyncArray<float_type> &y_p) const {
    CHECK_EQ(num_class * y.size(), y_p.size()) << num_class << " * " << y.size() << " != " << y_p.size();
    int n_instances = y.size();
    auto y_data = y.device_data();
    auto yp_data = y_p.device_data();
    SyncArray<int> is_true(n_instances);
    auto is_true_data = is_true.device_data();
    int num_class = this->num_class;
    device_loop(n_instances, [=] __device__(int i){
        int max_k = 0;
        float_type max_p = yp_data[i];
        for (int k = 1; k < num_class; ++k) {
            if (yp_data[k * n_instances + i] > max_p) {
                max_p = yp_data[k * n_instances + i];
                max_k = k;
            }
        }
        is_true_data[i] = max_k == y_data[i];
    });

    float acc = thrust::reduce(thrust::cuda::par, is_true_data, is_true_data + n_instances) / (float) n_instances;
    return acc;
}

float_type BinaryClassMetric::get_score(const SyncArray<float_type> &y_p) const {
    int n_instances = y.size();
    auto y_data = y.device_data();
    auto yp_data = y_p.device_data();
    SyncArray<int> is_true(n_instances);
    auto is_true_data = is_true.device_data();
    device_loop(n_instances, [=] __device__(int i){
        int max_k = (1 / (1 + exp(-yp_data[i])) > 0.5) ? 1 : 0;
        is_true_data[i] = max_k == y_data[i];
    });

    float acc = thrust::reduce(thrust::cuda::par, is_true_data, is_true_data + n_instances) / (float) n_instances;
    return 1 - acc;
}
