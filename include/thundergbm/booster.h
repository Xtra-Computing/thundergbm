//
// Created by ss on 19-1-17.
//

#ifndef THUNDERGBM_BOOSTER_H
#define THUNDERGBM_BOOSTER_H

#include <thundergbm/objective/objective_function.h>
#include <thundergbm/metric/metric.h>
#include <thundergbm/builder/function_builder.h>
#include <thundergbm/util/multi_device.h>
#include "thundergbm/common.h"
#include "syncarray.h"
#include "tree.h"
#include "row_sampler.h"

std::mutex mtx;

class Booster {
public:
    void init(const DataSet &dataSet, GBMParam &param);

    void boost(vector<vector<Tree>> &boosted_model, int epoch,int total_epoch);

private:
    MSyncArray<GHPair> gradients;
    std::unique_ptr<ObjectiveFunction> obj;
    std::unique_ptr<Metric> metric;
    MSyncArray<float_type> y;
    std::unique_ptr<FunctionBuilder> fbuilder;
    RowSampler rowSampler;
    GBMParam param;
    int n_devices;
};

void Booster::init(const DataSet &dataSet, GBMParam &param) {
    int n_available_device;
    cudaGetDeviceCount(&n_available_device);
    CHECK_GE(n_available_device, param.n_device) << "only " << n_available_device
                                            << " GPUs available; please set correct number of GPUs to use";
    this->param = param;
    //fbuilder.reset(FunctionBuilder::create(param.tree_method));
    //if method is hist, and n_available_device is 1
    if(param.n_device==1 && param.tree_method == "hist"){
        fbuilder.reset(FunctionBuilder::create("hist_single"));
    }
    else{
        fbuilder.reset(FunctionBuilder::create(param.tree_method));
    }

    fbuilder->init(dataSet, param);
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(param, dataSet);

    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients = MSyncArray<GHPair>(n_devices, n_outputs);
    y = MSyncArray<float_type>(n_devices, dataSet.n_instances());

    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
        y[device_id].copy_from(dataSet.y.data(), dataSet.n_instances());
    });

    //init base score
    //only support histogram-based method and single device now
    //TODO support exact and multi-device
    if(param.n_device && param.tree_method == "hist"){
        DO_ON_MULTI_DEVICES(n_devices, [&](int device_id){
            param.base_score = obj->init_base_score(y[device_id], fbuilder->get_raw_y_predict()[device_id], gradients[device_id]);
        });
    }
}

void Booster::boost(vector<vector<Tree>> &boosted_model,int epoch,int total_epoch) {
    TIMED_FUNC(timerObj);
    std::unique_lock<std::mutex> lock(mtx);

    //update gradients
    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
        obj->get_gradient(y[device_id], fbuilder->get_y_predict()[device_id], gradients[device_id]);
    });
    if (param.bagging) rowSampler.do_bagging(gradients);
    PERFORMANCE_CHECKPOINT(timerObj);
    //build new model/approximate function
    boosted_model.push_back(fbuilder->build_approximate(gradients));

    PERFORMANCE_CHECKPOINT(timerObj);
    //show metric on training set
    auto res =  metric->get_score(fbuilder->get_y_predict().front());
    LOG(INFO) <<"["<<epoch<<"/"<<total_epoch<<"] "<< metric->get_name() << " = " <<res;
}

#endif //THUNDERGBM_BOOSTER_H
