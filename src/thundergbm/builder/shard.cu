//
// Created by shijiashuai on 2019-03-08.
//
#include "thundergbm/builder/shard.h"
#include "thrust/sequence.h"
#include "thundergbm/util/device_lambda.cuh"

void Shard::column_sampling(float rate) {
    if (rate < 1) {
        CHECK_GT(rate, 0);
        int n_column = columns.n_column;
        SyncArray<int> idx(n_column);
        thrust::sequence(thrust::cuda::par, idx.device_data(), idx.device_end(), 0);
        std::random_shuffle(idx.host_data(), idx.host_data() + n_column);
        int sample_count = max(1, int(n_column * rate));
        ignored_set.resize(n_column);
        auto idx_data = idx.device_data();
        auto ignored_set_data = ignored_set.device_data();
        device_loop(sample_count, [=]__device__(int i) {
            ignored_set_data[idx_data[i]] = true;
        });
    }
}
