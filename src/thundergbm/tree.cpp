//
// Created by jiashuai on 18-1-18.
//
#include "thundergbm/tree.h"

Tree::Tree(int depth) {
    init(depth);
}

void Tree::init(int depth) {
    int n_max_nodes = static_cast<int>(pow(2, depth + 1) - 1);
    nodes.resize(n_max_nodes);
    TreeNode *node_data = nodes.host_data();
    for (int i = 0; i < n_max_nodes; ++i) {
        node_data[i].nid = i;
        node_data[i].fid = -1;
        node_data[i].is_valid = false;
    }
}

string Tree::to_string(int depth) {
    string s("\n");
    preorder_traversal(nodes.host_data()[0], depth, 0, s);
    return s;
}

void Tree::preorder_traversal(TreeNode &node, int max_depth, int depth, string &s) {
    if (node.is_valid)
        s = s + string(static_cast<unsigned long>(depth), '\t') +
            (node.is_leaf ?
             string_format("%d:leaf=%f\n", node.nid, node.base_weight) :
             string_format("%d:[f%d<%f], weight=%f, gain=%f, dr=%d\n", node.nid, node.fid + 1, node.split_value,
                           node.base_weight, node.gain, node.default_right));
    if (depth < max_depth) {
        preorder_traversal(nodes.host_data()[node.nid * 2 + 1], max_depth, depth + 1, s);
        preorder_traversal(nodes.host_data()[node.nid * 2 + 2], max_depth, depth + 1, s);
    }
}

std::ostream &operator<<(std::ostream &os, const Tree::TreeNode &node) {
    os << string_format("\nnid:%d,l:%d,fid:%d,f/i:%f/%d,gain:%.2f,r:%d,w:%f,", node.nid, node.is_leaf,
                        node.fid, node.split_value, node.split_index, node.gain, node.default_right, node.base_weight);
    os << "g/h:" << node.sum_gh_pair;
    return os;
}
