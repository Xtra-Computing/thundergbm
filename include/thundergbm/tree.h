//
// Created by jiashuai on 18-1-18.
//

#ifndef THUNDERGBM_TREE_H
#define THUNDERGBM_TREE_H

#include "thundergbm/thundergbm.h"
#include "syncarray.h"
#include "sstream"

class GHPair {
public:
    float_type g;
    float_type h;

    HOST_DEVICE GHPair operator+(const GHPair &rhs) const {
        GHPair res;
        res.g = this->g + rhs.g;
        res.h = this->h + rhs.h;
        return res;
    }

    HOST_DEVICE const GHPair operator-(const GHPair &rhs) const {
        GHPair res;
        res.g = this->g - rhs.g;
        res.h = this->h - rhs.h;
        return res;
    }

    HOST_DEVICE GHPair() : g(0), h(0) {};

    HOST_DEVICE GHPair(float_type v) : g(v), h(v) {};

    HOST_DEVICE GHPair(float_type g, float_type h) : g(g), h(h) {};

    friend std::ostream &operator<<(std::ostream &os,
                                    const GHPair &p) {
        os << string_format("%.2f/%.2f", p.g, p.h);
        return os;
    }
};

///instance statistics
class InsStat {
public:

    ///gradient and hessian
    SyncArray<GHPair> gh_pair;
    ///node id
    SyncArray<int> nid;
    ///target value
    SyncArray<float_type> y;
    ///predict value
    SyncArray<float_type> y_predict;

    size_t n_instances;

    InsStat() = default;

    explicit InsStat(size_t n_instances) {
        init(n_instances);
    }

    void init(size_t n_instances) {
        this->n_instances = n_instances;
        gh_pair.resize(n_instances);
        nid.resize(n_instances);
        y.resize(n_instances);
        y_predict.resize(n_instances);
    }
};

class Tree {
public:
    struct TreeNode {
        ///feature id
        int col_id;
        ///node id
        int nid;
        float_type gain;
        float_type base_weight;
        float_type split_value;
//        int split_index;
        bool default_right;
        bool is_leaf;
        bool is_valid;
        bool is_pruned;

//        unsigned int n_instances;

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

    explicit Tree(int depth);

    Tree() = default;

    Tree(const Tree &tree) {
        nodes.resize(tree.nodes.size());
        nodes.copy_from(tree.nodes);
    }

    void init(int depth);

    string to_string(int depth) const;

    void reorder_nid();

    SyncArray<Tree::TreeNode> nodes;
private:
    void preorder_traversal(int nid, int max_depth, int depth, string &s) const;
};

#endif //THUNDERGBM_TREE_H
