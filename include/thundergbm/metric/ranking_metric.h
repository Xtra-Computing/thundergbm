//
// Created by ss on 19-1-13.
//

#ifndef THUNDERGBM_RANKING_METRIC_H
#define THUNDERGBM_RANKING_METRIC_H

#include "metric.h"

class RankListMetric : public Metric {
public:
    float_type get_score(const SyncArray<float_type> &y_p) const override {
        float_type sum_score = 0;
        for (int k = 0; k < n_group; ++k) {
            int group_start = gptr[k];
            int len = gptr[k + 1] - group_start;
            SyncArray<float_type> query_y(len);
            SyncArray<float_type> query_yp(len);
            memcpy(query_y.host_data(), y.host_data() + group_start, len * sizeof(float_type));
            memcpy(query_yp.host_data(), y_p.host_data() + group_start, len * sizeof(float_type));
            sum_score += this->evalQuery(query_y, query_yp);
        }
        return sum_score / n_group;
    }

    void configure(const GBMParam &param, const DataSet &dataset) override {
        Metric::configure(param, dataset);

        //init gptr
        n_group = dataset.group.size();
        gptr = vector<int>(n_group + 1, 0);
        for (int i = 1; i < gptr.size(); ++i) {
            gptr[i] = gptr[i - 1] + dataset.group[i - 1];
        }

        //TODO parse from param
        topn = 2;
    }

protected:
    virtual float_type evalQuery(SyncArray<float_type> &y, SyncArray<float_type> &y_p) const = 0;

    vector<int> gptr;
    int n_group;
    int topn;
};

class MAP : public RankListMetric {
public:
    string get_name() const override {
        return "MAP";
    }

protected:
    float_type evalQuery(SyncArray<float_type> &y, SyncArray<float_type> &y_p) const override {
        auto y_data = y.host_data();
        auto yp_data = y_p.host_data();
        int len = y.size();
        std::sort(y_data, y_data + len, [=](int a, int b) { return yp_data[a] > yp_data[b]; });
        int nhits = 0;
        double sum_ap = 0;
        for (int i = 0; i < len; ++i) {
            if (y_data[i] != 0) {
                nhits++;
                if (i < topn) {
                    sum_ap += (double) nhits / (i + 1);
                }
            }
        }

        if (nhits != 0)
            return sum_ap / nhits;
        else return 0;
    };
};

#endif //THUNDERGBM_RANKING_METRIC_H
