//
// Created by ss on 19-1-12.
//
#include "thundergbm/objective/ranking_obj.h"

void LambdaRank::configure(GBMParam param, const DataSet &dataset) {
    sigma = 1;

    //init gptr
    n_group = dataset.group.size();
    gptr = vector<int>(n_group + 1, 0);
    for (int i = 1; i < gptr.size(); ++i) {
        gptr[i] = gptr[i - 1] + dataset.group[i-1];
    }

    LOG(DEBUG)<<gptr;

    idcg.resize(n_group);
    //calculate IDCG
    for (int k = 0; k < n_group; ++k) {
        int group_start = gptr[k];
        int len = gptr[k + 1] - group_start;
        vector<float_type> sorted_label(len);
        memcpy(sorted_label.data(), dataset.y.data() + group_start, len * sizeof(float_type));
        std::sort(sorted_label.begin(), sorted_label.end(), std::greater<float_type>());
        for (int i = 0; i < sorted_label.size(); ++i) {
            //assume labels are int
            idcg[k] += ((1 << (int) sorted_label[i]) - 1) / log2f(i + 1 + 1);
        }
    }
}

void
LambdaRank::get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p, SyncArray<GHPair> &gh_pair) {
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
                    float_type abs_delta_z = 1;
                    float_type rhoIJ = 1 / (1 + expf(sigma * (score[ui] - score[uj])));
                    float_type lambda = -abs_delta_z * rhoIJ;
                    float_type hessian = fmaxf(sigma * sigma * abs_delta_z * rhoIJ * (1 - rhoIJ), 1e-9f);
                    gh_data[ui].g += lambda;
                    gh_data[ui].h += 2 * hessian;
                    gh_data[uj].g -= lambda;
                    gh_data[uj].h += 2 * hessian;
                }
            }
        }
    }
}