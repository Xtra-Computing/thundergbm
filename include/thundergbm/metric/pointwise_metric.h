//
// Created by ss on 19-1-13.
//

#ifndef THUNDERGBM_POINTWISE_METRIC_H
#define THUNDERGBM_POINTWISE_METRIC_H

#include "metric.h"
#include "thrust/reduce.h"
#include "thundergbm/util/device_lambda.cuh"

class RMSE : public Metric {
public:
    float_type get_score(const SyncArray<float_type> &y_p) const override {
        CHECK_EQ(y_p.size(), y.size());
        int n_instances = y_p.size();
        SyncArray<float_type> sq_err(n_instances);
        auto sq_err_data = sq_err.device_data();
        const float_type *y_data = y.device_data();
        const float_type *y_predict_data = y_p.device_data();
        device_loop(n_instances, [=]
        __device__(int
        i) {
            float_type e = y_predict_data[i] - y_data[i];
            sq_err_data[i] = e * e;
        });
        float_type rmse =
                sqrt(thrust::reduce(thrust::cuda::par, sq_err.device_data(), sq_err.device_end()) / n_instances);
        return rmse;
    }

    string get_name() const override {
        return "RMSE";
    }
};

#endif //THUNDERGBM_POINTWISE_METRIC_H
