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
    unsigned char max_num_bin = 255;
    struct Shard {
        HistCut cut;
        SyncArray<int> bin_id;
        SyncArray<unsigned char> dense_bin_id;
        SparseColumns columns;
        InsStat stats;
        Tree tree;
        SyncArray<SplitPoint> sp;
        SyncArray<GHPair> last_hist;
        bool has_split;
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


    void get_bin_ids(const SparseColumns &columns, const HistCut &cut, SyncArray<int> &bin_id);

    void init_dense_data(const SparseColumns &columns, int n_instances, SyncArray<unsigned char> &dense_bin_id,
                             const SyncArray<int> &bin_id);

    void find_split(int level, const SparseColumns &columns, const Tree &tree, const InsStat &stats,
                        const HistCut &cut, SyncArray<SplitPoint> &sp,
                        const SyncArray<unsigned char> &dense_bin_id, SyncArray<GHPair> &last_hist);

    bool reset_ins2node_id(InsStat &stats, const Tree &tree, const SparseColumns &columns,
                               const SyncArray<unsigned char> &dense_bin_id);

    void split_point_all_reduce(SyncArray<SplitPoint> &global_sp, int depth);

    void update_tree(Tree &tree, const SyncArray<SplitPoint> &sp);
};

#endif //GBM_MIRROR2_HIST_UPDATER_H
