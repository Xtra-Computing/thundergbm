//
// Created by shijiashuai on 5/7/18.
//

#ifndef THUNDERGBM_EXACT_UPDATER_H
#define THUNDERGBM_EXACT_UPDATER_H

#include <thundergbm/tree.h>
#include <thundergbm/sparse_columns.h>
#include "thrust/reduce.h"
#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thundergbm/param.h>
#include "thundergbm/util/device_lambda.cuh"


class SplitPoint {
public:
    float_type gain;
    int split_fea_id;
    float_type fval;
    GHPair fea_missing_gh;
    GHPair rch_sum_gh;
    bool default_right;
    int nid;

    SplitPoint() {
        nid = -1;
        split_fea_id = -1;
        gain = std::numeric_limits<float_type>::min();
    }

    friend std::ostream &operator<<(std::ostream &output, const SplitPoint &sp) {
        output << sp.gain << "/" << sp.split_fea_id << "/" << sp.nid;
        return output;
    }
};

class ExactUpdater {
public:
    explicit ExactUpdater(GBMParam &param) {
        depth = param.depth;
        min_child_weight = param.min_child_weight;
        lambda = param.lambda;
        gamma = param.gamma;
        rt_eps = param.rt_eps;
        n_devices = param.n_device;
    }


    void grow(Tree &tree, const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats);

    int depth;
    float_type min_child_weight;
    float_type lambda;
    float_type gamma;
    float_type rt_eps;

    int n_devices;
    vector<std::shared_ptr<InsStat>> v_stats;
    vector<std::shared_ptr<Tree>> v_trees_gpu;

    void init_tree(Tree &tree, const InsStat &stats);

    virtual void find_split(int level, const SparseColumns &columns, const Tree &tree, const InsStat &stats,
                            SyncArray<SplitPoint> &sp);

//update global best split for each node
    void update_tree(Tree &tree, const SyncArray<SplitPoint> &sp);

    bool reset_ins2node_id(InsStat &stats, const Tree &tree, const SparseColumns &columns);

    void split_point_all_reduce(const vector<SyncArray<SplitPoint>> &local_sp, SyncArray<SplitPoint> &global_sp,
                                int depth);
};

typedef thrust::tuple<int, float_type> int_float;

std::ostream &operator<<(std::ostream &os, const int_float &rhs);

#endif //THUNDERGBM_EXACT_UPDATER_H
