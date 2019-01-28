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
    vector<Tree> build_approximate(const MSyncArray<GHPair> &gradients) override;

    void init(const DataSet &dataset, const GBMParam &param) override;

//    struct InternalShard: public Shard {

        void get_bin_ids(const SparseColumns &columns);

        void find_split(int level, int device_id) override;

        void update_ins2node_id() override;
//    };

    vector<Shard> shards;

//    void split_point_all_reduce(int depth);

//    void ins2node_id_all_reduce(int depth);

private:
    vector<HistCut> cut;
    MSyncArray<unsigned char> dense_bin_id;
    MSyncArray<GHPair> last_hist;
};


#endif //THUNDERGBM_HIST_TREE_BUILDER_H
