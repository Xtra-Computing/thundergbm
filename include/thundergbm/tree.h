//
// Created by jiashuai on 18-1-18.
//

#ifndef THUNDERGBM_TREE_H
#define THUNDERGBM_TREE_H

#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include "syncarray.h"
#include "sstream"
#include "ins_stat.h"


class Tree {
public:
    struct TreeNode {
        int final_id;// node id after pruning, may not equal to node index
        int lch_index;// index of left child
        int rch_index;// index of right child
        int parent_index;// index of parent node
        float_type gain;// gain of splitting this node
        float_type base_weight;
        int split_feature_id;
        float_type split_value;
        unsigned char split_bid;
        bool default_right;
        bool is_leaf;
        bool is_valid;// non-valid nodes are those that are "children" of leaf nodes
        bool is_pruned;// pruned after pruning

        GHPair sum_gh_pair;

        friend std::ostream &operator<<(std::ostream &os,
                                        const TreeNode &node);

        HOST_DEVICE void calc_weight(float_type lambda) {
            this->base_weight = -sum_gh_pair.g / (sum_gh_pair.h + lambda);
        }

        HOST_DEVICE bool splittable() const {
            return !is_leaf && is_valid;
        }

    };

    Tree() = default;

    Tree(const Tree &tree) {
        nodes.resize(tree.nodes.size());
        nodes.copy_from(tree.nodes);
    }

    Tree &operator=(const Tree &tree) {
        nodes.resize(tree.nodes.size());
        nodes.copy_from(tree.nodes);
        return *this;
    }

    void init(const InsStat &stats, const GBMParam &param);

    string dump(int depth) const;

    SyncArray<Tree::TreeNode> nodes;

    void prune_self(float_type gamma);

    //store the tree to file
private:
    friend class boost::serialization::access;
    template<class Archive>
    void process_nodes(Archive &ar){
        Tree::TreeNode *node_data = nodes.host_data();
        for(int nid = 0; nid < nodes.size(); nid++) {
            ar & node_data[nid].final_id;
            ar & node_data[nid].lch_index;
            ar & node_data[nid].rch_index;
            ar & node_data[nid].parent_index;
            ar & node_data[nid].gain;
            ar & node_data[nid].base_weight;
            ar & node_data[nid].split_feature_id;
            ar & node_data[nid].split_value;
            ar & node_data[nid].split_bid;
            ar & node_data[nid].default_right;
            ar & node_data[nid].is_leaf;
            ar & node_data[nid].is_valid;
            ar & node_data[nid].is_pruned;
            ar & node_data[nid].sum_gh_pair.g;
            ar & node_data[nid].sum_gh_pair.h;
        }
    }
    template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        size_t node_size = nodes.size();
        ar & node_size;
        const Tree::TreeNode *node_data = nodes.host_data();
        for(int nid = 0; nid < nodes.size(); nid++) {
            ar & node_data[nid].final_id;
            ar & node_data[nid].lch_index;
            ar & node_data[nid].rch_index;
            ar & node_data[nid].parent_index;
            ar & node_data[nid].gain;
            ar & node_data[nid].base_weight;
            ar & node_data[nid].split_feature_id;
            ar & node_data[nid].split_value;
            ar & node_data[nid].split_bid;
            ar & node_data[nid].default_right;
            ar & node_data[nid].is_leaf;
            ar & node_data[nid].is_valid;
            ar & node_data[nid].is_pruned;
            ar & node_data[nid].sum_gh_pair.g;
            ar & node_data[nid].sum_gh_pair.h;
        }
    }

    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        size_t node_size;
        ar & node_size;
        nodes.resize(node_size);
        process_nodes(ar);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

private:
    void preorder_traversal(int nid, int max_depth, int depth, string &s) const;

    int try_prune_leaf(int nid, int np, float_type gamma, vector<int> &leaf_child_count);

    void reorder_nid();
};

#endif //THUNDERGBM_TREE_H
