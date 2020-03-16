#include "gtest/gtest.h"
#include "thundergbm/parser.h"
#include "thundergbm/dataset.h"
#include "thundergbm/tree.h"

class ParserTest: public ::testing::Test {
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
};

TEST_F(ParserTest, test_parser){
    EXPECT_EQ(param.depth, 6);
    EXPECT_EQ(param.gamma, 1);
    EXPECT_EQ(param.learning_rate, 1);
    EXPECT_EQ(param.num_class, 1);
    EXPECT_EQ(param.tree_method, "auto");
    EXPECT_EQ(param.max_num_bin, 255);
}

TEST_F(ParserTest, test_save_model) {
    string model_path = "tgbm.model";
    vector<vector<Tree>> boosted_model;
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    Parser parser;
    parser.save_model(model_path, param, boosted_model, dataset);
}

TEST_F(ParserTest, test_load_model) {
    string model_path = "tgbm.model";
    vector<vector<Tree>> boosted_model;
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    Parser parser;
    parser.load_model(model_path, param, boosted_model, dataset);
    // the size of he boosted model should be zero
    EXPECT_EQ(boosted_model.size(), 0);
}
