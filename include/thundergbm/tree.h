//
// Created by jiashuai on 18-1-18.
//

#ifndef THUNDERGBM_TREE_H
#define THUNDERGBM_TREE_H

#include "thundergbm/thundergbm.h"
#include "syncarray.h"

///instance statistics
class InsStat {
public:
    ///gradient for each instance
    float_type g;
    ///hessian for each instance
    float_type h;
    ///node id for each instance
    int nid;
    ///target of each instance
    float_type y;

    friend std::ostream &operator<<(std::ostream &os,
                                    const InsStat &is) {
        os << string_format("\ng:%.2f,h:%.2f,nid:%d,y:%.2f", is.g, is.h, is.nid, is.y);
        return os;
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

        friend std::ostream &operator<<(std::ostream &os,
                                        const TreeNode &node) {
            os << string_format("\nnid:%d,fid:%d,loss:%.2f,base_weight:%.2f,#ins:%d", node.nid, node.fid,
                                node.loss, node.base_weight, node.n_instances);
            return os;
        }
    };

    explicit Tree(int depth);

    SyncArray<Tree::TreeNode> nodes;
};

#endif //THUNDERGBM_TREE_H
