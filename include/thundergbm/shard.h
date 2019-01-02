//
// Created by ss on 19-1-2.
//

#ifndef THUNDERGBM_SHARD_H
#define THUNDERGBM_SHARD_H


#include "param.h"
#include "ins_stat.h"
#include "sparse_columns.h"
#include "tree.h"

class SplitPoint;

struct Shard {
    GBMParam param;
    InsStat stats;
    Tree tree;
    SparseColumns columns;
    SyncArray<bool> ignored_set;//for column sampling
    SyncArray<SplitPoint> sp;
    bool has_split;

    void update_tree();

    void predict_in_training();

    void column_sampling();
};


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
        output << sp.gain << "/" << sp.split_fea_id << "/" << sp.nid << "/" << sp.rch_sum_gh;
        return output;
    }
};

#endif //THUNDERGBM_SHARD_H
