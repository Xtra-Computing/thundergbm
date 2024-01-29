//
// Created by ss on 19-1-1.
//

#ifndef THUNDERGBM_REGRESSION_OBJ_H
#define THUNDERGBM_REGRESSION_OBJ_H

#include "objective_function.h"
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/reduce.h"

template<template<typename> class Loss>
class RegressionObj : public ObjectiveFunction {
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override {
        CHECK_EQ(y.size(), y_p.size())<<y.size() << "!=" << y_p.size();
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
    
    //base score
    float init_base_score(const SyncArray<float_type> &y,SyncArray<float_type> &y_p, SyncArray<GHPair> &gh_pair){ 

        //get gradients first, SyncArray<GHPair> &gh_pair for temporal storage
        get_gradient(y,y_p,gh_pair);

        //get sum gh_pair
        GHPair sum_gh = thrust::reduce(thrust::cuda::par, gh_pair.device_data(), gh_pair.device_end());

        //get weight
        float weight =  -sum_gh.g / fmax(sum_gh.h, (double)(1e-6));
        
        float base_score = weight; 
        LOG(INFO)<<"base_score "<<base_score;
        auto y_p_data = y_p.device_data();
        device_loop(y_p.size(), [=]__device__(int i) {
            y_p_data[i] = base_score;
        });
        return base_score;
    }

    void configure(GBMParam param, const DataSet &dataset) override {}

    virtual ~RegressionObj() override = default;

    string default_metric_name() override {
        return "rmse";
    }
};

template<template<typename> class Loss>
class LogClsObj: public RegressionObj<Loss>{
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override {
        auto y_data = y.device_data();
        auto y_p_data = y_p.device_data();
        auto gh_pair_data = gh_pair.device_data();
        device_loop(y.size(), [=]__device__(int i) {
            gh_pair_data[i] = Loss<float_type>::gradient(y_data[i], y_p_data[i]);
        });
    }
    void predict_transform(SyncArray<float_type> &y) {
        //this method transform y(#class * #instances) into y(#instances)
        auto yp_data = y.device_data();
        auto label_data = label.device_data();
        // int num_class = this->num_class;
        int n_instances = y.size();
        device_loop(n_instances, [=]__device__(int i) {
            //yp_data[i] = Loss<float_type>::predict_transform(yp_data[i]);
            int max_k = (yp_data[i] > 0) ? 1 : 0;
            yp_data[i] = label_data[max_k];
        });
        //TODO not to make a temp_y?
        SyncArray < float_type > temp_y(n_instances);
        temp_y.copy_from(y.device_data(), n_instances);
        y.resize(n_instances);
        y.copy_from(temp_y);
    }

    //base score
    float init_base_score(const SyncArray<float_type> &y,SyncArray<float_type> &y_p, SyncArray<GHPair> &gh_pair){ 

        //get gradients first, SyncArray<GHPair> &gh_pair for temporal storage
        get_gradient(y,y_p,gh_pair);

        //get sum gh_pair
        GHPair sum_gh = thrust::reduce(thrust::cuda::par, gh_pair.device_data(), gh_pair.device_end());

        //get weight
        float weight =  -sum_gh.g / fmax(sum_gh.h, (double)(1e-6));
        //sigmod transform
        weight = 1 / (1 + expf(-weight));
        float base_score = -logf(1.0f / weight - 1.0f);
        LOG(INFO)<<"base_score "<<base_score;
        auto y_p_data = y_p.device_data();
        device_loop(y_p.size(), [=]__device__(int i) {
            y_p_data[i] = base_score;
        });
        return base_score;
    }
    string default_metric_name() override{
        return "error";
    }
    void configure(GBMParam param, const DataSet &dataset) {
        num_class = param.num_class;
        label.resize(num_class);
        CHECK_EQ(dataset.label.size(), num_class)<<dataset.label.size() << "!=" << num_class;
        label.copy_from(dataset.label.data(), num_class);
    }
protected:
    int num_class;
    SyncArray<float_type> label;
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
    HOST_DEVICE static GHPair gradient(float y, float y_p) {
        float p = sigmoid(y_p);
        return GHPair(p - y, fmaxf(p * (1 - p), 1e-16f));
    }

    HOST_DEVICE static float predict_transform(float y) { return sigmoid(y); }

    HOST_DEVICE static float sigmoid(float x) {return 1 / (1 + expf(-x));}
};

template<>
struct LogisticLoss<double> {
    HOST_DEVICE static GHPair gradient(double y, double y_p) {
        double p = sigmoid(y_p);
        return GHPair(p - y, fmax(p * (1 - p), 1e-16));
    }

    HOST_DEVICE static double predict_transform(double x) { return 1 / (1 + exp(-x)); }

    HOST_DEVICE static double sigmoid(double x) {return 1 / (1 + exp(-x));}
};

#endif //THUNDERGBM_REGRESSION_OBJ_H
