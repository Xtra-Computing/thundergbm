//
// Created by qinbin on 2018/7/6.
//

#ifndef GBM_MIRROR2_HIST_UPDATER_H
#define GBM_MIRROR2_HIST_UPDATER_H

#include "thundergbm/updater/exact_updater.h"
#include "thundergbm/hist_cut.h"

class HistUpdater : public ExactUpdater{
public:
    int max_num_bin = 256;
    int do_cut = 0;
    vector<HistCut> v_cut;
    vector<std::shared_ptr<SyncArray<int>>> bin_id;

    virtual void grow(Tree &tree, const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats);

    void init_cut(const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats, int n_instance);

    void get_bin_ids(const SparseColumns &columns);

    void find_split(int level, const SparseColumns &columns, const Tree &tree, const InsStat &stats, const HistCut &cut,
                    SyncArray<SplitPoint> &sp);

    explicit HistUpdater(GBMParam &param): ExactUpdater(param) {};
};
#endif //GBM_MIRROR2_HIST_UPDATER_H
