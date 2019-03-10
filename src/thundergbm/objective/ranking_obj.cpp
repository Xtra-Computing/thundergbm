//
// Created by ss on 19-1-12.
//
#include <thundergbm/objective/ranking_obj.h>
#include "thundergbm/metric/ranking_metric.h"
#ifndef _WIN32
#include <parallel/algorithm>
#endif
#include <random>

void LambdaRank::configure(GBMParam param, const DataSet &dataset) {
    sigma = 1;

    //init gptr
    n_group = dataset.group.size();
    RankListMetric::configure_gptr(dataset.group, gptr);
    CHECK_EQ(gptr.back(), dataset.n_instances());
}

void
LambdaRank::get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p, SyncArray<GHPair> &gh_pair) {
    TIMED_FUNC(obj);
    {
        auto gh_data = gh_pair.host_data();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < gh_pair.size(); ++i) {
            gh_data[i] = 0;
        }
    }
    GHPair *gh_data0 = gh_pair.host_data();
    const float_type *score0 = y_p.host_data();
    const float_type *label_data0 = y.host_data();
    PERFORMANCE_CHECKPOINT_WITH_ID(obj, "copy and init");
#pragma omp parallel for schedule(static)
    for (int k = 0; k < n_group; ++k) {
        int group_start = gptr[k];
        int len = gptr[k + 1] - group_start;
        GHPair *gh_data = gh_data0 + group_start;
        const float_type *score = score0 + group_start;
        const float_type *label_data = label_data0 + group_start;
        vector<int> idx(len);
        for (int i = 0; i < len; ++i) { idx[i] = i; }
        std::sort(idx.begin(), idx.end(), [=](int a, int b) { return score[a] > score[b]; });
        vector<std::pair<float_type, int>> label_idx(len);
        for (int i = 0; i < len; ++i) {
            label_idx[i].first = label_data[idx[i]];
            label_idx[i].second = idx[i];
        }
        //sort by label ascending
        std::sort(label_idx.begin(), label_idx.end(),
                  [](std::pair<float_type, int> a, std::pair<float_type, int> b) { return a.first > b.first; });

        std::mt19937 gen(std::rand());
        for (int i = 0; i < len; ++i) {
            int j = i + 1;
            while (j < len && label_idx[i].first == label_idx[j].first) j++;
            int nleft = i;
            int nright = len - j;
            //if not all are same label
            if (nleft + nright != 0) {
                // bucket in [i,j), get a sample outside bucket
                std::uniform_int_distribution<> dis(0, nleft + nright - 1);
                for (int l = i; l < j; ++l) {
                    int m = dis(gen);
                    int high = 0;
                    int low = 0;
                    if (m < nleft) {
                        high = m;
                        low = l;
                    } else {
                        high = l;
                        low = m + j - i;
                    }
                    float_type high_label = label_idx[high].first;
                    float_type low_label = label_idx[low].first;
                    int high_idx = label_idx[high].second;
                    int low_idx = label_idx[low].second;

                    float_type abs_delta_z = fabsf(get_delta_z(high_label, low_label, high, low, k));
                    float_type rhoIJ = 1 / (1 + expf((score[high_idx] - score[low_idx])));
                    float_type lambda = -abs_delta_z * rhoIJ;
                    float_type hessian = 2 * fmaxf(abs_delta_z * rhoIJ * (1 - rhoIJ), 1e-16f);
                    gh_data[high_idx] = gh_data[high_idx] + GHPair(lambda, hessian);
                    gh_data[low_idx] = gh_data[low_idx] + GHPair(-lambda, hessian);
                }
            }
            i = j;
        }
    }
    y_p.to_device();
}

string LambdaRank::default_metric_name() { return "map"; }

//inline functions should be defined in the header file
//inline float_type
//LambdaRank::get_delta_z(float_type labelI, float_type labelJ, int rankI, int rankJ, int group_id) { return 1; }


void LambdaRankNDCG::configure(GBMParam param, const DataSet &dataset) {
    LambdaRank::configure(param, dataset);
    NDCG::get_IDCG(gptr, dataset.y, idcg);
}

float_type
LambdaRankNDCG::get_delta_z(float_type labelI, float_type labelJ, int rankI, int rankJ, int group_id) {
    if (idcg[group_id] == 0) return 0;
    float_type dgI1 = NDCG::discounted_gain(static_cast<int>(labelI), rankI);
    float_type dgJ1 = NDCG::discounted_gain(static_cast<int>(labelJ), rankJ);
    float_type dgI2 = NDCG::discounted_gain(static_cast<int>(labelI), rankJ);
    float_type dgJ2 = NDCG::discounted_gain(static_cast<int>(labelJ), rankI);
    return (dgI1 + dgJ1 - dgI2 - dgJ2) / idcg[group_id];
}

string LambdaRankNDCG::default_metric_name() { return "ndcg"; }
