//
// Created by ss on 19-1-13.
//

#ifndef THUNDERGBM_POINTWISE_METRIC_H
#define THUNDERGBM_POINTWISE_METRIC_H

#include "metric.h"

class RMSE : public Metric {
public:
    float_type get_score(const SyncArray<float_type> &y_p) const override;

    string get_name() const override { return "RMSE"; }
};

class Accuracy: public Metric {
public:
    float_type get_score(const SyncArray<float_type> &y_p) const override;

    string get_name() const override;
};


#endif //THUNDERGBM_POINTWISE_METRIC_H
