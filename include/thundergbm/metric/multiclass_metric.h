//
// Created by ss on 19-1-15.
//

#ifndef THUNDERGBM_MULTICLASS_METRIC_H
#define THUNDERGBM_MULTICLASS_METRIC_H

#include "thundergbm/common.h"
#include "metric.h"

class MulticlassMetric: public Metric {
public:
    void configure(const GBMParam &param, const DataSet &dataset) override {
        Metric::configure(param, dataset);
        num_class = param.num_class;
        CHECK_EQ(num_class, dataset.label.size());
        label.resize(num_class);
        label.copy_from(dataset.label.data(), num_class);
    }

protected:
    int num_class;
    SyncArray<float_type> label;
};

class MulticlassAccuracy: public MulticlassMetric {
public:
    float_type get_score(const SyncArray<float_type> &y_p) const override;

    string get_name() const override { return "multi-class accuracy"; }
};

class BinaryClassMetric: public MulticlassAccuracy{
public:
    float_type get_score(const SyncArray<float_type> &y_p) const override;
    string get_name() const override { return "test error";}
};


#endif //THUNDERGBM_MULTICLASS_METRIC_H
