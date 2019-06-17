//
// Created by jiashuai on 17-9-17.
//
#include "gtest/gtest.h"
#include "thundergbm/dataset.h"

class DatasetTest : public ::testing::Test {
public:
    GBMParam param;
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

TEST_F(DatasetTest, load_dataset){
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    printf("### Dataset: %s, num_instances: %d, num_features: %d, get_cut_points finished. ###\n",
           param.path.c_str(),
           dataset.n_instances(),
           dataset.n_features());
    EXPECT_EQ(dataset.n_instances(), 1605);
    EXPECT_EQ(dataset.n_features_, 119);
    EXPECT_EQ(dataset.label[0], -1);
    EXPECT_EQ(dataset.csr_val[1], 1);
}

