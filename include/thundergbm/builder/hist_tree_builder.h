//
// Created by ss on 19-1-17.
//

#ifndef THUNDERGBM_HIST_TREE_BUILDER_H
#define THUNDERGBM_HIST_TREE_BUILDER_H

#include <thundergbm/hist_cut.h>
#include "thundergbm/common.h"
#include "shard.h"
#include "tree_builder.h"


class HistTreeBuilder : public TreeBuilder {
public:

    void init(const DataSet &dataset, const GBMParam &param) override;

    void get_bin_ids();

    void find_split(int level, int device_id) override;

    void update_ins2node_id() override;

    vector<Shard> shards;

private:
    vector<HistCut> cut;
    MSyncArray<unsigned char> dense_bin_id;
    MSyncArray<GHPair> last_hist;
};


#endif //THUNDERGBM_HIST_TREE_BUILDER_H
