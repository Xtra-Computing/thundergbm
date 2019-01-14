//
// Created by shijiashuai on 5/7/18.
//

#ifndef THUNDERGBM_EXACT_UPDATER_H
#define THUNDERGBM_EXACT_UPDATER_H

#include <thundergbm/tree.h>
#include <thundergbm/sparse_columns.h>
#include "thundergbm/shard.h"
#include "thundergbm/util/multi_device.h"


class ExactUpdater {
public:
    explicit ExactUpdater(GBMParam &param) {
        this->param = param;
    }

    GBMParam param;

    struct ExactShard : public Shard {
        void find_split(int level);

        void update_ins2node_id();
    };

    vector<std::unique_ptr<ExactShard>> shards;

    template<typename L>
    void for_each_shard(L lambda) {
        DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id) {
            lambda(*shards[device_id].get());
        });
    }

    void init(const DataSet &dataset);

    void grow(Tree &tree);


    void split_point_all_reduce(int depth);

    void ins2node_id_all_reduce(int depth);
};

#endif //THUNDERGBM_EXACT_UPDATER_H
