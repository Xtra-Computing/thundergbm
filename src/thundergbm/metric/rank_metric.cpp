//
// Created by ss on 19-1-14.
//
#include <thundergbm/metric/ranking_metric.h>
#ifndef _WIN32
#include "parallel/algorithm"
#endif

float_type RankListMetric::get_score(const SyncArray<float_type> &y_p) const {
    TIMED_FUNC(obj);
    float_type sum_score = 0;
    auto y_data0 = y.host_data();
    auto yp_data0 = y_p.host_data();
#pragma omp parallel for schedule(static) reduction(+:sum_score)
    for (int k = 0; k < n_group; ++k) {
        int group_start = gptr[k];
        int len = gptr[k + 1] - group_start;
        vector<float_type> query_y(len);
        vector<float_type> query_yp(len);
        memcpy(query_y.data(), y_data0 + group_start, len * sizeof(float_type));
        memcpy(query_yp.data(), yp_data0 + group_start, len * sizeof(float_type));
        sum_score += this->eval_query_group(query_y, query_yp, k);
    }
    return sum_score / n_group;
}

void RankListMetric::configure(const GBMParam &param, const DataSet &dataset) {
    Metric::configure(param, dataset);

    //init gptr
    n_group = dataset.group.size();
    configure_gptr(dataset.group, gptr);

    //TODO parse from param
    topn = (std::numeric_limits<int>::max)();
}

void RankListMetric::configure_gptr(const vector<int> &group, vector<int> &gptr) {
    gptr = vector<int>(group.size() + 1, 0);
    for (int i = 1; i < gptr.size(); ++i) {
        gptr[i] = gptr[i - 1] + group[i - 1];
    }
}

float_type MAP::eval_query_group(vector<float_type> &y, vector<float_type> &y_p, int group_id) const {
    auto y_data = y.data();
    auto yp_data = y_p.data();
    int len = y.size();
    vector<int> idx(len);
    for (int i = 0; i < len; ++i) {
        idx[i] = i;
    }
#ifdef _WIN32
	std::sort(idx.begin(), idx.end(), [=](int a, int b) { return yp_data[a] > yp_data[b]; });
#else
    __gnu_parallel::sort(idx.begin(), idx.end(), [=](int a, int b) { return yp_data[a] > yp_data[b]; });
#endif
    int nhits = 0;
    double sum_ap = 0;
    for (int i = 0; i < len; ++i) {
        if (y_data[idx[i]] != 0) {
            nhits++;
            if (i < topn) {
                sum_ap += (double) nhits / (i + 1);
            }
        }
    }

    if (nhits != 0)
        return sum_ap / nhits;
    else return 1;
}

void NDCG::configure(const GBMParam &param, const DataSet &dataset) {
    RankListMetric::configure(param, dataset);
    get_IDCG(gptr, dataset.y, idcg);
}

float_type NDCG::eval_query_group(vector<float_type> &y, vector<float_type> &y_p, int group_id) const {
    CHECK_EQ(y.size(), y_p.size());
    if (idcg[group_id] == 0) return 1;
    int len = y.size();
    vector<int> idx(len);
    for (int i = 0; i < len; ++i) {
        idx[i] = i;
    }
    auto label = y.data();
    auto score = y_p.data();
#ifdef _WIN32
	std::sort(idx.begin(), idx.end(), [=](int a, int b) { return score[a] > score[b]; });
#else
    __gnu_parallel::sort(idx.begin(), idx.end(), [=](int a, int b) { return score[a] > score[b]; });
#endif

    float_type dcg = 0;
    for (int i = 0; i < len; ++i) {
        dcg += discounted_gain(static_cast<int>(label[idx[i]]), i);
    }
    return dcg / idcg[group_id];
}

void NDCG::get_IDCG(const vector<int> &gptr, const vector<float_type> &y, vector<float_type> &idcg) {
    int n_group = gptr.size() - 1;
    idcg.clear();
    idcg.resize(n_group);
    //calculate IDCG
#pragma omp parallel for schedule(static)
    for (int k = 0; k < n_group; ++k) {
        int group_start = gptr[k];
        int len = gptr[k + 1] - group_start;
        vector<float_type> sorted_label(len);
        memcpy(sorted_label.data(), y.data() + group_start, len * sizeof(float_type));
#ifdef _WIN32
		std::sort(sorted_label.begin(), sorted_label.end(), std::greater<float_type>());
#else
        __gnu_parallel::sort(sorted_label.begin(), sorted_label.end(), std::greater<float_type>());
#endif
        for (int i = 0; i < sorted_label.size(); ++i) {
            //assume labels are int
            idcg[k] += NDCG::discounted_gain(static_cast<int>(sorted_label[i]), i);
        }
    }
}
