//
// Created by jiashuai on 19-1-23.
//

#ifndef THUNDERGBM_TREEBUILDER_H
#define THUNDERGBM_TREEBUILDER_H

#include <thundergbm/tree.h>
#include "thundergbm/common.h"
#include "shard.h"
#include "function_builder.h"


class TreeBuilder : public FunctionBuilder {
public:
    virtual void find_split(int level, int device_id) = 0;

    virtual void update_ins2node_id() = 0;

    //new func
    virtual void update_ins2node_id(int level) = 0;

    vector<Tree> build_approximate(const MSyncArray<GHPair> &gradients) override;

    void init(const DataSet &dataset, const GBMParam &param) override;

    virtual void update_tree();

    void predict_in_training(int k);

    virtual void split_point_all_reduce(int depth);

    virtual void ins2node_id_all_reduce(int depth);

    virtual ~TreeBuilder(){};

protected:
    vector<Shard> shards;
    int n_instances;
    vector<Tree> trees;
    MSyncArray<int> ins2node_id;
    MSyncArray<SplitPoint> sp;
    MSyncArray<GHPair> gradients;
    vector<bool> has_split;
};


#endif //THUNDERGBM_TREEBUILDER_H
