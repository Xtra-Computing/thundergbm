#include "gtest/gtest.h"
#include "thundergbm/trainer.h"
#include "thundergbm/predictor.h"
#include "thundergbm/dataset.h"
#include "thundergbm/tree.h"

class GBDTTest: public ::testing::Test {
public:
    GBMParam param;
protected:
    void SetUp() override {
        param.depth = 3;
        param.n_trees = 5;
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
        param.tree_per_rounds = 1;
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

// training GBDTs in hist manner
TEST_F(GBDTTest, test_hist) {
    param.tree_method = "hist";
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    TreeTrainer trainer;
    trainer.train(param, dataset);
}

// training GBDTs in exact manner
TEST_F(GBDTTest, test_exact) {
    param.tree_method = "exact";
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    TreeTrainer trainer;
    trainer.train(param, dataset);
}

// test bagging
 TEST_F(GBDTTest,test_bagging) {
    param.bagging = true;
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    TreeTrainer trainer;
    trainer.train(param, dataset);
 }

// test different number of parallel trees
TEST_F(GBDTTest, test_n_parallel_trees) {
    param.n_parallel_trees = 2;
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    TreeTrainer trainer;
    trainer.train(param, dataset);
}

// test predictor
TEST_F(GBDTTest, test_predictor) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    TreeTrainer trainer;
    vector<vector<Tree>> boosted_model;
    boosted_model = trainer.train(param, dataset);
    Predictor predictor;
    predictor.predict(param, boosted_model, dataset);
}
