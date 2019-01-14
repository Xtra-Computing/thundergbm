//
// Created by ss on 19-1-12.
//
#include <thundergbm/objective/ranking_obj.h>
#include "thundergbm/metric/ranking_metric.h"

void LambdaRank::configure(GBMParam param, const DataSet &dataset) {
    sigma = 1;

    //init gptr
    n_group = dataset.group.size();
    RankListMetric::configure_gptr(dataset.group, gptr);
}

void
LambdaRank::get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p, SyncArray<GHPair> &gh_pair) {
    {
        auto gh_data = gh_pair.host_data();
        for (int i = 0; i < gh_pair.size(); ++i) {
            gh_data[i] = 0;
        }
    }
    for (int k = 0; k < n_group; ++k) {
        int group_start = gptr[k];
        int len = gptr[k + 1] - group_start;
        GHPair *gh_data = gh_pair.host_data() + group_start;
        const float_type *score = y_p.host_data() + group_start;
        const float_type *label_data = y.host_data() + group_start;
        vector<int> idx(len);
        for (int i = 0; i < len; ++i) { idx[i] = i; }
        std::sort(idx.begin(), idx.end(), [=](int a, int b) { return score[a] > score[b]; });

        for (int i = 0; i < len; ++i) {
            int ui = idx[i];
            for (int j = 0; j < len; ++j) {
                int uj = idx[j];
                // for every pair Ui > Uj, calculate lambdaIJ, and increase lambdaI, decrease lambdaJ
                // by it respectively
                if (label_data[ui] > label_data[uj]) {
                    float_type abs_delta_z = fabsf(get_delta_z(label_data[ui], label_data[uj], i, j, k));
                    float_type rhoIJ = 1 / (1 + expf(sigma * (score[ui] - score[uj])));
                    float_type lambda = -abs_delta_z * rhoIJ;
                    float_type hessian = fmaxf(sigma * sigma * abs_delta_z * rhoIJ * (1 - rhoIJ), 1e-16f);
                    gh_data[ui].g += lambda;
                    gh_data[ui].h += 2 * hessian;
                    gh_data[uj].g -= lambda;
                    gh_data[uj].h += 2 * hessian;
                }
            }
        }
    }
}

string LambdaRank::default_metric() { return "map"; }

float_type
LambdaRank::get_delta_z(float_type labelI, float_type labelJ, int rankI, int rankJ, int group_id) { return 1; }


void LambdaRankNDCG::configure(GBMParam param, const DataSet &dataset) {
    LambdaRank::configure(param, dataset);
    NDCG::get_IDCG(gptr, dataset.y, idcg);
}

float_type LambdaRankNDCG::get_delta_z(float_type labelI, float_type labelJ, int rankI, int rankJ, int group_id) {
    if (idcg[group_id] == 0) return 0;
    float_type dgI1 = NDCG::discounted_gain(static_cast<int>(labelI), rankI);
    float_type dgJ1 = NDCG::discounted_gain(static_cast<int>(labelJ), rankJ);
    float_type dgI2 = NDCG::discounted_gain(static_cast<int>(labelI), rankJ);
    float_type dgJ2 = NDCG::discounted_gain(static_cast<int>(labelJ), rankI);
    return (dgI1 + dgJ1 - dgI2 - dgJ2) / idcg[group_id];
}

string LambdaRankNDCG::default_metric() { return "ndcg"; }
