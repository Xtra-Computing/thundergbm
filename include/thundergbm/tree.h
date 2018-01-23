//
// Created by jiashuai on 18-1-18.
//

#ifndef THUNDERGBM_TREE_H
#define THUNDERGBM_TREE_H

#include "thundergbm/thundergbm.h"
#include "syncarray.h"

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
        int fid;
        ///node id
        int nid;
        float_type loss;
        float_type base_weight;

        unsigned int n_instances;

        GHPair sum_gh_pair;

        TreeNode() : fid(0), nid(0), loss(0), base_weight(0), n_instances(0), sum_gh_pair() {};

        friend std::ostream &operator<<(std::ostream &os,
                                        const TreeNode &node) {
            os << string_format("\nnid:%d,fid:%d,loss:%.2f,base_weight:%.2f,#ins:%d,sg:%.2f,sh:%.2f", node.nid,
                                node.fid, node.loss, node.base_weight, node.n_instances, node.sum_gh_pair.g,
                                node.sum_gh_pair.h);
            return os;
        }
    };

    explicit Tree(int depth);

    Tree() = default;

    void init(int depth);

    SyncArray<Tree::TreeNode> nodes;
};

#endif //THUNDERGBM_TREE_H
