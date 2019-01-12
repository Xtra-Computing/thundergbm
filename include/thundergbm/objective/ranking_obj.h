//
// Created by ss on 19-1-10.
//

#ifndef THUNDERGBM_RANKING_OBJ_H
#define THUNDERGBM_RANKING_OBJ_H

#include "objective_function.h"
#include "thundergbm/util/device_lambda.cuh"

/**
 *
 * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
 */
class LambdaRank : public ObjectiveFunction {
public:
    void get_gradient(const SyncArray<float_type> &y, const SyncArray<float_type> &y_p,
                      SyncArray<GHPair> &gh_pair) override {
        SyncArray<int> label(y_p.size());
        {
            auto y_data = y.device_data();
            auto label_data = label.device_data();
            //labels are all int
            device_loop(y.size(), [=]__device__(int i) {
                label_data[i] = (int) y_data[i];
            });
        }
        //calculate IDCG
        vector<float_type> idcg(n_group);
        for (int k = 0; k < n_group; ++k) {
            int group_start = gptr[k];
            int len = gptr[k + 1] - group_start;
            vector<int> sorted_label(len);
            memcpy(sorted_label.data(), label.host_data() + group_start, len * sizeof(int));
            std::sort(sorted_label.begin(), sorted_label.end(), std::greater<int>());
            for (int i = 0; i < sorted_label.size(); ++i) {
                idcg[k] += ((1 << sorted_label[i]) - 1) / log2f(i + 1 + 1);
            }
        }

        for (int k = 0; k < n_group; ++k) {
            int group_start = gptr[k];
            int len = gptr[k + 1] - group_start;
            GHPair *gh_data = gh_pair.host_data() + group_start;
            const float_type *score = y_p.host_data() + group_start;
            const int *label_data = label.host_data() + group_start;
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
                        gh_data[ui] = gh_data[ui] + GHPair(lambda, hessian);
                        gh_data[uj] = gh_data[uj] - GHPair(lambda, hessian);
                    }
                }
            }
        }
    }

    void configure(GBMParam param) override {
        //init gptr
        sigma = 1;
    }

    virtual

    ~LambdaRank() override = default;

private:
    vector<int> gptr;//group start position
    int n_group;

    float_type sigma;
};

#endif //THUNDERGBM_RANKING_OBJ_H
