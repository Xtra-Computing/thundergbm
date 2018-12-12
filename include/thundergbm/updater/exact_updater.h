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
#include "thrust/iterator/discard_iterator.h"
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
    unsigned char split_bid;

    SplitPoint() {
        nid = -1;
        split_fea_id = -1;
        gain = 0;
    }

    friend std::ostream &operator<<(std::ostream &output, const SplitPoint &sp) {
        output << sp.gain << "/" << sp.split_fea_id << "/" << sp.nid << "/" <<sp.rch_sum_gh;
        return output;
    }
};

class ExactUpdater {
public:
    explicit ExactUpdater(GBMParam &param) {
        this->param = param;
    }


    GBMParam param;
    struct Shard {
        GBMParam param;
        SparseColumns columns;
        InsStat stats;
        Tree tree;
        SyncArray<SplitPoint> sp;
        bool has_split;
        void find_split(int level);

        void update_tree();

        void reset_ins2node_id();
        void predict_in_training();

    };
    vector<std::unique_ptr<Shard>> shards;
    
    template <typename L>
    void for_each_shard(L lambda){
        DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
            lambda(*shards[device_id].get());
        });
    }
    
    void init(const DataSet& dataset);

    void grow(Tree &tree);


    void split_point_all_reduce(int depth);
    void ins2node_id_all_reduce(int depth);
};

typedef thrust::tuple<int, float_type> int_float;

std::ostream &operator<<(std::ostream &os, const int_float &rhs);

#endif //THUNDERGBM_EXACT_UPDATER_H
