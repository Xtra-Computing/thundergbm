//
// Created by ss on 19-1-2.
//
#include "thundergbm/thundergbm.h"
#include "thundergbm/shard.h"
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/reduce.h"

void Shard::update_tree() {
    TIMED_FUNC(timerObj);
    auto sp_data = sp.device_data();
    LOG(DEBUG) << sp;
    int n_nodes_in_level = sp.size();

    Tree::TreeNode *nodes_data = tree.nodes.device_data();
    float_type rt_eps = param.rt_eps;
    float_type lambda = param.lambda;

//    LOG(DEBUG) << n_nodes_in_level;
    device_loop(n_nodes_in_level, [=]__device__(int i) {
        float_type best_split_gain = sp_data[i].gain;
        if (best_split_gain > rt_eps) {
            //do split
            if (sp_data[i].nid == -1) return;
            int nid = sp_data[i].nid;
            Tree::TreeNode &node = nodes_data[nid];
            node.gain = best_split_gain;

            Tree::TreeNode &lch = nodes_data[node.lch_index];//left child
            Tree::TreeNode &rch = nodes_data[node.rch_index];//right child
            lch.is_valid = true;
            rch.is_valid = true;
            node.split_feature_id = sp_data[i].split_fea_id;
            GHPair p_missing_gh = sp_data[i].fea_missing_gh;
            //todo process begin
            node.split_value = sp_data[i].fval;
            node.split_bid = sp_data[i].split_bid;
            rch.sum_gh_pair = sp_data[i].rch_sum_gh;
            if (sp_data[i].default_right) {
                rch.sum_gh_pair = rch.sum_gh_pair + p_missing_gh;
                node.default_right = true;
            }
            lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
            lch.calc_weight(lambda);
            rch.calc_weight(lambda);
        } else {
            //set leaf
            if (sp_data[i].nid == -1) return;
            int nid = sp_data[i].nid;
            Tree::TreeNode &node = nodes_data[nid];
            node.is_leaf = true;
            nodes_data[node.lch_index].is_valid = false;
            nodes_data[node.rch_index].is_valid = false;
        }
//    }
    });
    LOG(DEBUG) << tree.nodes;
}

void Shard::predict_in_training() {
    auto y_predict_data = stats.y_predict.device_data();
    auto nid_data = stats.nid.device_data();
    const Tree::TreeNode *nodes_data = tree.nodes.device_data();
    auto lr = param.learning_rate;
    device_loop(stats.n_instances, [=]__device__(int i) {
        int nid = nid_data[i];
        while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
        y_predict_data[i] += lr * nodes_data[nid].base_weight;
    });
}

void Shard::column_sampling() {
    if (param.column_sampling_rate < 1) {
        CHECK_GT(param.column_sampling_rate, 0);
        int n_column = columns.n_column;
        SyncArray<int> idx(n_column);
        thrust::sequence(thrust::cuda::par, idx.device_data(), idx.device_end(), 0);
        std::random_shuffle(idx.host_data(), idx.host_data() +n_column);
        int sample_count = max(1, int(n_column * param.column_sampling_rate));
        ignored_set.resize(n_column);
        auto idx_data = idx.device_data();
        auto ignored_set_data = ignored_set.device_data();
        device_loop(sample_count, [=]__device__(int i) {
            ignored_set_data[idx_data[i]] = true;
        });
    }
}
