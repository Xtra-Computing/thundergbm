//
// Created by ss on 19-1-1.
//

#ifndef THUNDERGBM_REGRESSION_OBJ_H
#define THUNDERGBM_REGRESSION_OBJ_H

#include "objective_function.h"
#include "thundergbm/util/device_lambda.cuh"

template<template<typename> class Loss>
class RegressionObj : public ObjectiveFunction {
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override {
        CHECK_EQ(y.size(), y_p.size());
        CHECK_EQ(y.size(), gh_pair.size());
        auto y_data = y.device_data();
        auto y_p_data = y_p.device_data();
        auto gh_pair_data = gh_pair.device_data();
        device_loop(y.size(), [=]__device__(int i) {
            gh_pair_data[i] = Loss<float_type>::gradient(y_data[i], y_p_data[i]);
        });
    }

    void predict_transform(SyncArray<float_type> &y) override {
        auto y_data = y.device_data();
        device_loop(y.size(), [=]__device__(int i) {
            y_data[i] = Loss<float_type>::predict_transform(y_data[i]);
        });
    }

    void configure(GBMParam param) override {}

    ~RegressionObj() override = default;
};

template<typename T>
struct SquareLoss {
    HOST_DEVICE static GHPair gradient(T y, T y_p) { return GHPair(y_p - y, 1); }

    HOST_DEVICE static T predict_transform(T x) { return x; }
};

//for probability regression
template<typename T>
struct LogisticLoss {
    HOST_DEVICE static GHPair gradient(T y, T y_p);

    HOST_DEVICE static T predict_transform(T x);
};

template<>
struct LogisticLoss<float> {
    HOST_DEVICE static GHPair gradient(float y, float y_p) { return GHPair(y_p - y, fmaxf(y_p * (1 - y_p), 1e-16f)); }

    HOST_DEVICE static float predict_transform(float x) { return 1 / (1 + expf(-x)); }
};

template<>
struct LogisticLoss<double> {
    HOST_DEVICE static GHPair gradient(double y, double y_p) { return GHPair(y_p - y, fmax(y_p * (1 - y_p), 1e-16)); }

    HOST_DEVICE static double predict_transform(double x) { return 1 / (1 + exp(-x)); }
};

#endif //THUNDERGBM_REGRESSION_OBJ_H
