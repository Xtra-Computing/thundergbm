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

    static void configure_gptr(const vector<int> &group, vector<int> &gptr);

protected:
    virtual float_type eval_query_group(vector<float_type> &y, vector<float_type> &y_p, int group_id) const = 0;

    vector<int> gptr;
    int n_group;
    int topn;
};


class MAP : public RankListMetric {
public:
    string get_name() const override { return "MAP"; }

protected:
    float_type eval_query_group(vector<float_type> &y, vector<float_type> &y_p, int group_id) const override;
};

class NDCG : public RankListMetric {
public:
    string get_name() const override { return "NDCG"; };

    void configure(const GBMParam &param, const DataSet &dataset) override;

    inline HOST_DEVICE static float_type discounted_gain(int label, int rank) {
        return ((1 << label) - 1) / log2f(rank + 1 + 1);
    }

    static void get_IDCG(const vector<int> &gptr, const vector<float_type> &y, vector<float_type> &idcg);

protected:
    float_type eval_query_group(vector<float_type> &y, vector<float_type> &y_p, int group_id) const override;

private:
    vector<float_type> idcg;
};


#endif //THUNDERGBM_RANKING_METRIC_H
