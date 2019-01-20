//
// Created by ss on 19-1-17.
//

#ifndef THUNDERGBM_HIST_TREE_BUILDER_H
#define THUNDERGBM_HIST_TREE_BUILDER_H

#include <thundergbm/hist_cut.h>
#include <thundergbm/util/multi_device.h>
#include "thundergbm/common.h"
#include "function_builder.h"
#include "shard.h"


class HistTreeBuilder : public FunctionBuilder {
public:
    vector<Tree> build_approximate(const MSyncArray<GHPair> &gradients) override;

    void init(const DataSet &dataset, const GBMParam &param) override;

    const MSyncArray<float_type>& get_y_predict() override;

    struct InternalShard: public Shard {
        HistCut cut;
        SyncArray<unsigned char> dense_bin_id;
        SyncArray<GHPair> last_hist;
        int rank;

        void get_bin_ids(const SparseColumns &columns);

        void find_split(int level);

        void update_ins2node_id();
    };

    template<typename L>
    static void for_each_shard(vector<InternalShard> &shards, L lambda) {
        DO_ON_MULTI_DEVICES(shards.size(), [&](int device_id) {
            lambda(shards[device_id]);
        });
    }

    vector<InternalShard> shards;

    void split_point_all_reduce(int depth, vector<InternalShard> &shards);

    void ins2node_id_all_reduce(vector<InternalShard> &shards);
    GBMParam param;
private:
    MSyncArray<float_type> y_predict;
};


#endif //THUNDERGBM_HIST_TREE_BUILDER_H
