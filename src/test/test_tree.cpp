#include "gtest/gtest.h"
#include "thundergbm/tree.h"
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
        param.tree_method = "auto";
        if (!param.verbose) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "True");
        }
        if (!param.profiling) {
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        }
    }
}


TEST_F(TreeTest, init){
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    Booster booster;
    booster.init(dataset, param);
    vector<vector<Tree>> boosted_model;
    for (int i = 0; i < param.n_trees; ++i) {
        booster.boost(boosted_model);
    }
    EXCEPT_EQ(boosted_model.size(), 1);
    EXCEPT_EQ(boosted_model[0].size(), 40);
}


TEST_F(TreeTest, treenode){
    int max_nodes = 8;
    Synarray<TreeNode> nodes;
    nodes = Synarray<TreeNode>(max_nodes);
    auto node_data = nodes.host_data();
    for(int i =0; i < max_nodes; i++) {
        node_data[i].final_id = i;
        node_data[i].split_feature_id = -1;
    }

    EXCEPT_EQ(nodes.size(), 8);
    EXCEPT_EQ(node_data[5].final_id, 5);
    EXCEPT_EQ(node_data[6].split_feature_id, -1);


