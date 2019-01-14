//
// Created by qinbin on 2018/7/6.
//

#ifndef GBM_MIRROR2_HIST_UPDATER_H
#define GBM_MIRROR2_HIST_UPDATER_H

#include "thundergbm/common.h"
#include "thundergbm/shard.h"
#include "thundergbm/hist_cut.h"
#include "thundergbm/sparse_columns.h"
#include "thundergbm/util/multi_device.h"

class HistUpdater {
public:
    struct InternalShard: public Shard {
        HistCut cut;
        SyncArray<unsigned char> dense_bin_id;
        SyncArray<GHPair> last_hist;
        int rank;

        void get_bin_ids(const SparseColumns &columns);

        void find_split(int level);

        void update_ins2node_id();
    };

    typedef InternalShard ShardT;
    GBMParam param;

    template<typename L>
    static void for_each_shard(vector<ShardT> &shards, L lambda) {
        DO_ON_MULTI_DEVICES(shards.size(), [&](int device_id) {
            lambda(shards[device_id]);
        });
    }

    explicit HistUpdater(GBMParam &param) {
        this->param = param;
    }

    void init(vector<ShardT> &shards);

    void grow(vector<Tree> &trees, vector<ShardT> &shards);

    void split_point_all_reduce(int depth, vector<ShardT> &shards);

    void ins2node_id_all_reduce(vector<ShardT> &shards);
};

#endif //GBM_MIRROR2_HIST_UPDATER_H
