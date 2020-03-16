#include "gtest/gtest.h"
#include "thundergbm/metric/metric.h"
#include "thundergbm/metric/multiclass_metric.h"
#include "thundergbm/metric/pointwise_metric.h"
#include "thundergbm/metric/ranking_metric.h"


class MetricTest : public ::testing::Test {
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


TEST_F(MetricTest, test_multiclass_metric_config) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    MulticlassAccuracy mmetric;
    dataset.y.resize(4);
    dataset.label.resize(2);
    mmetric.configure(param, dataset);
    EXPECT_EQ(mmetric.get_name(), "multi-class accuracy");
}

TEST_F(MetricTest, test_multiclass_metric_score) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    MulticlassAccuracy mmetric;
    dataset.y.resize(4);
    dataset.label.resize(2);
    mmetric.configure(param, dataset);

    SyncArray<float_type> y_pred(8);
    auto y_pred_data = y_pred.host_data();
    y_pred_data[0] = 1;
    EXPECT_EQ(mmetric.get_score(y_pred), 0) << mmetric.get_score(y_pred);
}

TEST_F(MetricTest, test_binaryclass_metric_score) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    BinaryClassMetric mmetric;
    dataset.y.resize(4);
    dataset.label.resize(2);
    mmetric.configure(param, dataset);

    SyncArray<float_type> y_pred(4);
    auto y_pred_data = y_pred.host_data();
    y_pred_data[0] = 1;
    EXPECT_EQ(mmetric.get_score(y_pred), 1) << mmetric.get_score(y_pred);
}


TEST_F(MetricTest, test_pointwise_metric_score) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    RMSE mmetric;
    dataset.y.resize(4);
    mmetric.configure(param, dataset);

    SyncArray<float_type> y_pred(4);
    EXPECT_EQ(mmetric.get_score(y_pred), 1) << mmetric.get_score(y_pred);
}
