//
// Created by qinbin on 2018/7/6.
//

#ifndef GBM_MIRROR2_HIST_UPDATER_H
#define GBM_MIRROR2_HIST_UPDATER_H

#include "thundergbm/updater/exact_updater.h"
#include "thundergbm/hist_cut.h"

class HistUpdater {
public:
    explicit HistUpdater(GBMParam &param) {
        this->param = param;
    }

    GBMParam param;

    struct Shard {
        GBMParam param;
        HistCut cut;
        SyncArray<unsigned char> dense_bin_id;
        InsStat stats;
        Tree tree;
        SyncArray<bool> ignored_set;
        SyncArray<SplitPoint> sp;
        SyncArray<GHPair> last_hist;
        bool has_split;
        int n_column;
        int column_offset;
        int idx;

        void get_bin_ids(const SparseColumns &columns);

        void find_split(int level);

        void reset_ins2node_id();

        void update_tree();

        void predict_in_training();

        void update_ignored_set();
    };

    vector<std::unique_ptr<Shard>> shards;

    template<typename L>
    void for_each_shard(L lambda) {
        DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id) {
            lambda(*shards[device_id].get());
        });
    }

    void init(const DataSet &dataset);

    void grow(Tree &tree);

    void split_point_all_reduce(int depth);

    void ins2node_id_all_reduce();
};

#endif //GBM_MIRROR2_HIST_UPDATER_H
