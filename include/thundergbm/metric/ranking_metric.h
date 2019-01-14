//
// Created by ss on 19-1-13.
//

#ifndef THUNDERGBM_RANKING_METRIC_H
#define THUNDERGBM_RANKING_METRIC_H

#include "metric.h"

class RankListMetric : public Metric {
public:
    float_type get_score(const SyncArray<float_type> &y_p) const override;

    void configure(const GBMParam &param, const DataSet &dataset) override;

protected:
    virtual float_type evalQuery(SyncArray<float_type> &y, SyncArray<float_type> &y_p) const = 0;

    vector<int> gptr;
    int n_group;
    int topn;
};


class MAP : public RankListMetric {
public:
    string get_name() const override { return "MAP"; }

protected:
    float_type evalQuery(SyncArray<float_type> &y, SyncArray<float_type> &y_p) const override;
};


#endif //THUNDERGBM_RANKING_METRIC_H
