#include "gtest/gtest.h"
#include "thundergbm/objective/multiclass_obj.h"
#include "thundergbm/objective/regression_obj.h"
#include "thundergbm/objective/ranking_obj.h"
#include "thundergbm/dataset.h"
#include "thundergbm/parser.h"
#include "thundergbm/syncarray.h"


class GradientTest: public ::testing::Test {
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
        param.num_class = 2;
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

TEST_F(GradientTest, test_softmax_obj) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    param.num_class = 2;
    dataset.label.resize(2);
    dataset.label[0] = 0;
    dataset.label[1] = 1;
    Softmax softmax;
    softmax.configure(param, dataset);
    
    // check the metric name
    EXPECT_EQ(softmax.default_metric_name(), "macc");
    SyncArray<float_type> y_true(4);
    SyncArray<float_type> y_pred(8);
    auto y_pred_data = y_pred.host_data();
    for(int i = 0; i < 8; i++)
        y_pred_data[i] = -i;
    SyncArray<GHPair> gh_pair(8);
    softmax.get_gradient(y_true, y_pred, gh_pair);

    // test the transform function
    EXPECT_EQ(y_pred.size(), 8);
    softmax.predict_transform(y_pred);
    EXPECT_EQ(y_pred.size(), 4);
}

TEST_F(GradientTest, test_softmaxprob_obj) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    param.num_class = 2;
    dataset.label.resize(2);
    dataset.label[0] = 0;
    dataset.label[1] = 1;
    SoftmaxProb smp;
    smp.configure(param, dataset);
    
    // check the metric name
    EXPECT_EQ(smp.default_metric_name(), "macc");
    SyncArray<float_type> y_true(4);
    SyncArray<float_type> y_pred(8);
    auto y_pred_data = y_pred.host_data();
    for(int i = 0; i < 8; i++)
        y_pred_data[i] = -i;
    SyncArray<GHPair> gh_pair(8);
    smp.get_gradient(y_true, y_pred, gh_pair);

    // test the transform function
    EXPECT_EQ(y_pred.size(), 8);
    smp.predict_transform(y_pred);
    EXPECT_EQ(y_pred.size(), 8);
}

TEST_F(GradientTest, test_regression_obj) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    RegressionObj<SquareLoss> rmse;
    SyncArray<float_type> y_true(4);
    SyncArray<float_type> y_pred(4);
    auto y_pred_data = y_pred.host_data();
    for(int i = 0; i < 4; i++)
        y_pred_data[i] = -i;
    SyncArray<GHPair> gh_pair(4);
    EXPECT_EQ(rmse.default_metric_name(), "rmse");
    rmse.get_gradient(y_true, y_pred, gh_pair);
}

TEST_F(GradientTest, test_logcls_obj) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    LogClsObj<SquareLoss> logcls;
    SyncArray<float_type> y_true(4);
    SyncArray<float_type> y_pred(4);
    auto y_pred_data = y_pred.host_data();
    for(int i = 0; i < 4; i++)
        y_pred_data[i] = -i;
    SyncArray<GHPair> gh_pair(4);
    EXPECT_EQ(logcls.default_metric_name(), "error");
    logcls.get_gradient(y_true, y_pred, gh_pair);
}

/*TEST_F(GradientTest, test_squareloss_obj) {*/
    /*DataSet dataset;*/
    /*dataset.load_from_file(param.path, param);*/
/*}*/
