//
// Created by ss on 19-1-19.
//

#ifndef THUNDERGBM_EXACT_TREE_BUILDER_H
#define THUNDERGBM_EXACT_TREE_BUILDER_H

#include "thundergbm/common.h"
#include "shard.h"
#include "tree_builder.h"


class ExactTreeBuilder : public TreeBuilder {
public:

    void init(const DataSet &dataset, const GBMParam &param) override;

    void find_split(int level, int device_id) override;

    void ins2node_id_all_reduce(int depth) override;

    void update_ins2node_id() override;

    virtual ~ExactTreeBuilder(){};

};


#endif //THUNDERGBM_EXACT_TREE_BUILDER_H
