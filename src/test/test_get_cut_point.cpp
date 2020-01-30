//
// Created by hanfeng on 6/11/19.
//

#include "gtest/gtest.h"
#include "thundergbm/common.h"
#include "thundergbm/sparse_columns.h"
#include "thundergbm/hist_cut.h"
#include "thrust/unique.h"
#include "thrust/execution_policy.h"


class GetCutPointTest : public ::testing::Test {
public:
    GBMParam param;
    HistCut cut;

    vector<float_type> cut_points;
    vector<int> row_ptr;
    //for gpu
    SyncArray<float_type> cut_points_val;
    SyncArray<int> cut_row_ptr;
    SyncArray<int> cut_fid;


protected:
    void SetUp() override {
        param.max_num_bin = 255;
        param.depth = 6;
        param.n_trees = 40;
        param.n_device = 1;
        param.min_child_weight = 1;
        param.lambda = 1;
        param.gamma = 1;
        param.rt_eps = 1e-6;
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
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
        }
        if (!param.profiling) {
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        }
    }

    void get_cut_points2(SparseColumns &columns, int max_num_bins, int n_instances){
        int n_column = columns.n_column;
        auto csc_val_data = columns.csc_val.host_data();
        auto csc_col_ptr_data = columns.csc_col_ptr.host_data();
        cut_points.clear();
        row_ptr.clear();
        row_ptr.resize(1, 0);

        //TODO do this on GPU
        for (int fid = 0; fid < n_column; ++fid) {
            int col_start = csc_col_ptr_data[fid];
            int col_len = csc_col_ptr_data[fid + 1] - col_start;
            auto val_data = csc_val_data + col_start;
            vector<float_type> unique_val(col_len);

            int unique_len = thrust::unique_copy(thrust::host, val_data, val_data + col_len, unique_val.data()) - unique_val.data();
            if (unique_len <= max_num_bins) {
                row_ptr.push_back(unique_len + row_ptr.back());
                for (int i = 0; i < unique_len; ++i) {
                    cut_points.push_back(unique_val[i]);
                }
            } else {
                row_ptr.push_back(max_num_bins + row_ptr.back());
                for (int i = 0; i < max_num_bins; ++i) {
                    cut_points.push_back(unique_val[unique_len / max_num_bins * i]);
                }
            }
        }

        cut_points_val.resize(cut_points.size());
        cut_points_val.copy_from(cut_points.data(), cut_points.size());
        cut_row_ptr.resize(row_ptr.size());
        cut_row_ptr.copy_from(row_ptr.data(), row_ptr.size());
        cut_fid.resize(cut_points.size());
    }
};


TEST_F(GetCutPointTest, covtype) {
    param.path = "../dataset/covtype";
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    SparseColumns columns;
//    vector<std::unique_ptr<SparseColumns>> v_columns(1);
//    columns.csr2csc_gpu(dataset, v_columns);
//    cut.get_cut_points2(columns, param.max_num_bin, dataset.n_instances());

    printf("### Dataset: %s, num_instances: %d, num_features: %d, get_cut_points finished. ###\n",
           param.path.c_str(),
           dataset.n_instances(),
           dataset.n_features());

    this->get_cut_points2(columns, param.max_num_bin,dataset.n_instances());

    // --- test cut_points_val
    auto gpu_cut_points_val = cut.cut_points_val.host_data();
    auto cpu_cut_points_val = this->cut_points_val.host_data();
    for(int i = 0; i < cut.cut_points_val.size(); i++)
        EXPECT_EQ(gpu_cut_points_val[i], cpu_cut_points_val[i]);

    // --- test cut_row_ptr
    auto gpu_cut_row_ptr = cut.cut_row_ptr.host_data();
    auto cpu_cut_row_ptr = this->cut_row_ptr.host_data();
    for(int i = 0; i < cut.cut_row_ptr.size(); i++)
        EXPECT_EQ(gpu_cut_row_ptr[i], cpu_cut_row_ptr[i]);

    // --- test cut_fid
    EXPECT_EQ(cut.cut_fid.size(), this->cut_fid.size());
}


TEST_F(GetCutPointTest, real_sim) {
    param.path = "../dataset/real-sim";
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    SparseColumns columns;
    vector<std::unique_ptr<SparseColumns>> v_columns(1);
    columns.csr2csc_gpu(dataset, v_columns);
    cut.get_cut_points2(columns, param.max_num_bin, dataset.n_instances());

    printf("### Dataset: %s, num_instances: %d, num_features: %d, get_cut_points finished. ###\n",
           param.path.c_str(),
           dataset.n_instances(),
           dataset.n_features());

    this->get_cut_points2(columns, param.max_num_bin,dataset.n_instances());

    // --- test cut_points_val
    auto gpu_cut_points_val = cut.cut_points_val.host_data();
    auto cpu_cut_points_val = this->cut_points_val.host_data();
    for(int i = 0; i < cut.cut_points_val.size(); i++)
        EXPECT_EQ(gpu_cut_points_val[i], cpu_cut_points_val[i]);

    // --- test cut_row_ptr
    auto gpu_cut_row_ptr = cut.cut_row_ptr.host_data();
    auto cpu_cut_row_ptr = this->cut_row_ptr.host_data();
    for(int i = 0; i < cut.cut_row_ptr.size(); i++)
        EXPECT_EQ(gpu_cut_row_ptr[i], cpu_cut_row_ptr[i]);

    // --- test cut_fid
    EXPECT_EQ(cut.cut_fid.size(), this->cut_fid.size());
}

TEST_F(GetCutPointTest, susy) {
    param.path = "../dataset/SUSY";
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    SparseColumns columns;
    vector<std::unique_ptr<SparseColumns>> v_columns(1);
    columns.csr2csc_gpu(dataset, v_columns);
    cut.get_cut_points2(columns, param.max_num_bin, dataset.n_instances());

    printf("### Dataset: %s, num_instances: %d, num_features: %d, get_cut_points finished. ###\n",
           param.path.c_str(),
           dataset.n_instances(),
           dataset.n_features());

    this->get_cut_points2(columns, param.max_num_bin,dataset.n_instances());

    // --- test cut_points_val
    auto gpu_cut_points_val = cut.cut_points_val.host_data();
    auto cpu_cut_points_val = this->cut_points_val.host_data();
    for(int i = 0; i < cut.cut_points_val.size(); i++)
        EXPECT_EQ(gpu_cut_points_val[i], cpu_cut_points_val[i]);

    // --- test cut_row_ptr
    auto gpu_cut_row_ptr = cut.cut_row_ptr.host_data();
    auto cpu_cut_row_ptr = this->cut_row_ptr.host_data();
    for(int i = 0; i < cut.cut_row_ptr.size(); i++)
        EXPECT_EQ(gpu_cut_row_ptr[i], cpu_cut_row_ptr[i]);

    // --- test cut_fid
    EXPECT_EQ(cut.cut_fid.size(), this->cut_fid.size());
}
