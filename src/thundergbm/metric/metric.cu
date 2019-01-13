//
// Created by ss on 19-1-13.
//
#include "thundergbm/metric/metric.h"
#include "thundergbm/metric/pointwise_metric.h"
#include "thundergbm/metric/ranking_metric.h"

Metric *Metric::create(string name) {
    if (name == "map") return new MAP;
    if (name == "rmse") return new RMSE;
    LOG(FATAL) << "unknown metric " << name;
    return nullptr;
}
