//
// Created by qinbin on 2018/7/6.
//

#ifndef GBM_MIRROR2_HIST_UPDATER_H
#define GBM_MIRROR2_HIST_UPDATER_H

#include "thundergbm/updater/exact_updater.h"
#include "thundergbm/hist_cut.h"

class HistUpdater : public ExactUpdater{
public:
    unsigned char max_num_bin = 255;
    int do_cut = 0;
    vector<HistCut> v_cut;
    vector<std::shared_ptr<SyncArray<int>>> bin_id;
    SyncArray<unsigned char> dense_bin_id;
    SyncArray<GHPair> last_hist;

    virtual void grow(Tree &tree, const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats);

    void init_cut(const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats, int n_instance);

    void get_bin_ids(const SparseColumns &columns);

    void init_dense_data(const SparseColumns &columns, int n_instances);

    void find_split(int level, const SparseColumns &columns, const Tree &tree, const InsStat &stats, const HistCut &cut,
                    SyncArray<SplitPoint> &sp);
    explicit HistUpdater(GBMParam &param): ExactUpdater(param) {};

    virtual bool reset_ins2node_id(InsStat &stats, const Tree &tree, const SparseColumns &columns);
};
#endif //GBM_MIRROR2_HIST_UPDATER_H
