#include "gtest/gtest.h"
#include "thundergbm/tree.h"
#include "thundergbm/dataset.h"
#include "thundergbm/booster.h"
#include "thundergbm/syncarray.h"

class TreeTest: public ::testing::Test {
public:
    GBMParam param;
    vector<float_type> csr_val;
    vector<int> csr_row_ptr;
    vector<int> csr_col_idx;
    vector<float_type> y;
    size_t n_features_;
    vector<float_type> label;
protected:
    void SetUp() override {
        param.depth = 6;
        param.n_trees = 40;
        param.n_device = 1;
        param.min_child_weight = 1;
        param.lambda = 1;
        param.gamma = 1;
        param.rt_eps = 1e-6;
        param.max_num_bin = 255;
        param.verbose = false;
        param.profiling = false;
        param.column_sampling_rate = 1;
        param.bagging = false;
        param.n_parallel_trees = 1;
        param.learning_rate = 1;
        param.objective = "reg:linear";
        param.num_class = 1;
        param.path = "../dataset/test_dataset.txt";
        param.tree_method = "hist";
        if (!param.verbose) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "True");
        }
        if (!param.profiling) {
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        }
    }
};

TEST_F(TreeTest, treenode){
    int max_nodes = 8;
    SyncArray<Tree::TreeNode> nodes;
    nodes = SyncArray<Tree::TreeNode>(max_nodes);
    auto node_data = nodes.host_data();
    for(int i =0; i < max_nodes; i++) {
        node_data[i].final_id = i;
        node_data[i].split_feature_id = -1;
    }

    EXPECT_EQ(nodes.size(), 8);
    EXPECT_EQ(node_data[5].final_id, 5);
    EXPECT_EQ(node_data[6].split_feature_id, -1);
}

TEST_F(TreeTest, tree_init){
    SyncArray<GHPair> gradients(10);
    Tree tree;
    tree.init2(gradients, param);

    // check the amount of tree nodes
    EXPECT_EQ(tree.nodes.size(), 127);

    // check the value of nodes' attributes
    auto nodes_data = tree.nodes.host_data();
    EXPECT_EQ(nodes_data[5].final_id, 5);
    EXPECT_EQ(nodes_data[1].split_feature_id, -1);
}

TEST_F(TreeTest, tree_prune) {
    SyncArray<GHPair> gradients(10);
    Tree tree;
    tree.init2(gradients, param);
    tree.prune_self(0.5);
}
