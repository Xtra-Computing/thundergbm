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
    }
}
