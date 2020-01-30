//
// Created by hanfeng on 6/11/19.
//

#include <thundergbm/parser.h>
#include <thundergbm/trainer.h>
#include "gtest/gtest.h"
#include "thundergbm/common.h"
#include "thundergbm/sparse_columns.h"
#include "thundergbm/builder/shard.h"
#include "cusparse.h"
#include "thundergbm/dataset.h"

class CSR2CSCTest : public ::testing::Test {
public:
    GBMParam param;

    SyncArray<float_type> csc_val;
    SyncArray<int> csc_row_idx;
    SyncArray<int> csc_col_ptr;

    SyncArray<float_type> csc_val2;
    SyncArray<int> csc_row_idx2;
    SyncArray<int> csc_col_ptr2;
    int n_column;
    int n_row;
    int column_offset;
    int nnz;
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
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
        }
        if (!param.profiling) {
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        }
    }

    void csr2csc(const DataSet &dataset) {
        LOG(TRACE) << "constructing sparse columns, converting csr to csc using CPU";
        //cpu transpose
        n_column = dataset.n_features();
        n_row = dataset.n_instances();
        nnz = dataset.csr_val.size();
        csc_val.resize(nnz);
        csc_row_idx.resize(nnz);
        csc_col_ptr.resize(n_column + 1);
        LOG(INFO) << string_format("#non-zeros = %ld, density = %.2f%%", nnz,
                                   (float) nnz / n_column / dataset.n_instances() * 100);
        auto csc_val_data = csc_val.host_data();
        auto csc_row_idx_data = csc_row_idx.host_data();
        auto csc_col_ptr_data = csc_col_ptr.host_data();
        for (int i = 0; i < nnz; ++i) {
            csc_col_ptr_data[dataset.csr_col_idx[i] + 1] += 1;
        }
        for (int i = 1; i < n_column + 1; ++i) {
            csc_col_ptr_data[i] += csc_col_ptr_data[i - 1];
        }
        for (int row = 0; row < dataset.n_instances(); ++row) {
            for (int j = dataset.csr_row_ptr[row]; j < dataset.csr_row_ptr[row + 1]; ++j) {
                int col = dataset.csr_col_idx[j]; // csr col
                int dest = csc_col_ptr_data[col]; // destination index in csc array
                csc_val_data[dest] = dataset.csr_val[j];
                csc_row_idx_data[dest] = row;
                csc_col_ptr_data[col]++; //increment column start position
            }
        }

        //recover column start position
        for (int i = 0, last = 0; i < n_column; ++i) {
            int next_last = csc_col_ptr_data[i];
            csc_col_ptr_data[i] = last;
            last = next_last;
        }
    }

    void csr2csc_gpu(const DataSet &dataset) {
        LOG(INFO) << "convert csr to csc using gpu...";
        //three arrays (on GPU/CPU) for csr representation
        SyncArray<float_type> val;
        SyncArray<int> col_idx;
        SyncArray<int> row_ptr;
        val.resize(dataset.csr_val.size());
        col_idx.resize(dataset.csr_col_idx.size());
        row_ptr.resize(dataset.csr_row_ptr.size());

        //copy data to the three arrays
        val.copy_from(dataset.csr_val.data(), val.size());
        col_idx.copy_from(dataset.csr_col_idx.data(), col_idx.size());
        row_ptr.copy_from(dataset.csr_row_ptr.data(), row_ptr.size());
        cusparseHandle_t handle;
        cusparseMatDescr_t descr;
        cusparseCreate(&handle);
        cusparseCreateMatDescr(&descr);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

        n_column = dataset.n_features_;
        n_row = dataset.n_instances();
        nnz = dataset.csr_val.size();
        csc_val2.resize(nnz);
        csc_row_idx2.resize(nnz);
        csc_col_ptr2.resize(n_column + 1);

        cusparseScsr2csc(handle, dataset.n_instances(), n_column, nnz, val.device_data(), row_ptr.device_data(),
                         col_idx.device_data(), csc_val2.device_data(), csc_row_idx2.device_data(), csc_col_ptr2.device_data(),
                         CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
        cudaDeviceSynchronize();
        cusparseDestroy(handle);
        cusparseDestroyMatDescr(descr);
    }

};


TEST_F(CSR2CSCTest, covtype) {
    param.path = "../dataset/covtype";
    DataSet dataset;
    dataset.load_from_file(param.path, param);

    printf("### Dataset: %s, num_instances: %d, num_features: %d, csr2csc finished. ###\n",
            param.path.c_str(),
            dataset.n_instances(),
            dataset.n_features());

    this->csr2csc(dataset);
    this->csr2csc_gpu(dataset);


    // --- test csc_val
    auto gpu_csc_val_data = this->csc_val2.host_data();
    auto cpu_csc_val_data = this->csc_val.host_data();
    for(int i = 0; i < this->csc_val2.size(); i++) {
        EXPECT_EQ(gpu_csc_val_data[i], cpu_csc_val_data[i]);
    }

    // --- test csc_row_idx
    auto gpu_csc_row_idx = this->csc_row_idx2.host_data();
    auto cpu_csc_row_idx = this->csc_row_idx.host_data();
    for(int i = 0; i < this->csc_row_idx2.size(); i++) {
        EXPECT_EQ(gpu_csc_row_idx[i], cpu_csc_row_idx[i]);
    }

    // --- test csc_col_ptr
    auto gpu_csc_col_ptr = this->csc_col_ptr2.host_data();
    auto cpu_csc_col_ptr = this->csc_col_ptr.host_data();
    for(int i = 0; i < this->csc_col_ptr2.size(); i++) {
        EXPECT_EQ(gpu_csc_col_ptr[i], cpu_csc_col_ptr[i]);
    }
}

TEST_F(CSR2CSCTest, e2006) {
    param.path = "../dataset/E2006";
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    SparseColumns columns;
    vector<std::unique_ptr<SparseColumns>> v_columns(1);
    columns.csr2csc_gpu(dataset, v_columns);
    printf("### Dataset: %s, num_instances: %d, num_features: %d, csr2csc finished. ###\n",
           param.path.c_str(),
           dataset.n_instances(),
           dataset.n_features());

    this->csr2csc(dataset);

    EXPECT_EQ(columns.n_column, this->n_column);
    EXPECT_EQ(columns.n_row, this->n_row);

    // --- test csc_val
    auto gpu_csc_val_data = columns.csc_val.host_data();
    auto cpu_csc_val_data = this->csc_val.host_data();
    for(int i = 0; i < columns.csc_val.size(); i++) {
        EXPECT_EQ(gpu_csc_val_data[i], cpu_csc_val_data[i]);
    }

    // --- test csc_row_idx
    auto gpu_csc_row_idx = columns.csc_row_idx.host_data();
    auto cpu_csc_row_idx = this->csc_row_idx.host_data();
    for(int i = 0; i < columns.csc_row_idx.size(); i++) {
        EXPECT_EQ(gpu_csc_row_idx[i], cpu_csc_row_idx[i]);
    }

    // --- test csc_col_ptr
    auto gpu_csc_col_ptr = columns.csc_col_ptr.host_data();
    auto cpu_csc_col_ptr = this->csc_col_ptr.host_data();
    for(int i = 0; i < columns.csc_col_ptr.size(); i++) {
        EXPECT_EQ(gpu_csc_col_ptr[i], cpu_csc_col_ptr[i]);
    }
}

TEST_F(CSR2CSCTest, higgs) {
    param.path = "../dataset/HIGGS";
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    SparseColumns columns;
    vector<std::unique_ptr<SparseColumns>> v_columns(1);
    columns.csr2csc_gpu(dataset, v_columns);
    printf("### Dataset: %s, num_instances: %d, num_features: %d, csr2csc finished. ###\n",
           param.path.c_str(),
           dataset.n_instances(),
           dataset.n_features());

    this->csr2csc(dataset);

    EXPECT_EQ(columns.n_column, this->n_column);
    EXPECT_EQ(columns.n_row, this->n_row);

    // --- test csc_val
    auto gpu_csc_val_data = columns.csc_val.host_data();
    auto cpu_csc_val_data = this->csc_val.host_data();
    for(int i = 0; i < columns.csc_val.size(); i++) {
        EXPECT_EQ(gpu_csc_val_data[i], cpu_csc_val_data[i]);
    }

    // --- test csc_row_idx
    auto gpu_csc_row_idx = columns.csc_row_idx.host_data();
    auto cpu_csc_row_idx = this->csc_row_idx.host_data();
    for(int i = 0; i < columns.csc_row_idx.size(); i++) {
        EXPECT_EQ(gpu_csc_row_idx[i], cpu_csc_row_idx[i]);
    }

    // --- test csc_col_ptr
    auto gpu_csc_col_ptr = columns.csc_col_ptr.host_data();
    auto cpu_csc_col_ptr = this->csc_col_ptr.host_data();
    for(int i = 0; i < columns.csc_col_ptr.size(); i++) {
        EXPECT_EQ(gpu_csc_col_ptr[i], cpu_csc_col_ptr[i]);
    }
}

TEST_F(CSR2CSCTest, real_sim) {
    param.path = "../dataset/real-sim";
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    SparseColumns columns;
    vector<std::unique_ptr<SparseColumns>> v_columns(1);
    columns.csr2csc_gpu(dataset, v_columns);
    printf("### Dataset: %s, num_instances: %d, num_features: %d, csr2csc finished. ###\n",
           param.path.c_str(),
           dataset.n_instances(),
           dataset.n_features());

    this->csr2csc(dataset);

    EXPECT_EQ(columns.n_column, this->n_column);
    EXPECT_EQ(columns.n_row, this->n_row);

    // --- test csc_val
    auto gpu_csc_val_data = columns.csc_val.host_data();
    auto cpu_csc_val_data = this->csc_val.host_data();
    for(int i = 0; i < columns.csc_val.size(); i++) {
        EXPECT_EQ(gpu_csc_val_data[i], cpu_csc_val_data[i]);
    }

    // --- test csc_row_idx
    auto gpu_csc_row_idx = columns.csc_row_idx.host_data();
    auto cpu_csc_row_idx = this->csc_row_idx.host_data();
    for(int i = 0; i < columns.csc_row_idx.size(); i++) {
        EXPECT_EQ(gpu_csc_row_idx[i], cpu_csc_row_idx[i]);
    }

    // --- test csc_col_ptr
    auto gpu_csc_col_ptr = columns.csc_col_ptr.host_data();
    auto cpu_csc_col_ptr = this->csc_col_ptr.host_data();
    for(int i = 0; i < columns.csc_col_ptr.size(); i++) {
        EXPECT_EQ(gpu_csc_col_ptr[i], cpu_csc_col_ptr[i]);
    }
}

TEST_F(CSR2CSCTest, susy) {
    param.path = "../dataset/SUSY";
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    SparseColumns columns;
    vector<std::unique_ptr<SparseColumns>> v_columns(1);
    columns.csr2csc_gpu(dataset, v_columns);
    printf("### Dataset: %s, num_instances: %d, num_features: %d, csr2csc finished. ###\n",
           param.path.c_str(),
           dataset.n_instances(),
           dataset.n_features());

    this->csr2csc(dataset);

    EXPECT_EQ(columns.n_column, this->n_column);
    EXPECT_EQ(columns.n_row, this->n_row);

    // --- test csc_val
    auto gpu_csc_val_data = columns.csc_val.host_data();
    auto cpu_csc_val_data = this->csc_val.host_data();
    for(int i = 0; i < columns.csc_val.size(); i++) {
        EXPECT_EQ(gpu_csc_val_data[i], cpu_csc_val_data[i]);
    }

    // --- test csc_row_idx
    auto gpu_csc_row_idx = columns.csc_row_idx.host_data();
    auto cpu_csc_row_idx = this->csc_row_idx.host_data();
    for(int i = 0; i < columns.csc_row_idx.size(); i++) {
        EXPECT_EQ(gpu_csc_row_idx[i], cpu_csc_row_idx[i]);
    }

    // --- test csc_col_ptr
    auto gpu_csc_col_ptr = columns.csc_col_ptr.host_data();
    auto cpu_csc_col_ptr = this->csc_col_ptr.host_data();
    for(int i = 0; i < columns.csc_col_ptr.size(); i++) {
        EXPECT_EQ(gpu_csc_col_ptr[i], cpu_csc_col_ptr[i]);
    }
}


TEST_F(CSR2CSCTest, log1p) {
    param.path = "../dataset/log1p";
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    SparseColumns columns;
    vector<std::unique_ptr<SparseColumns>> v_columns(1);
    columns.csr2csc_gpu(dataset, v_columns);
    printf("### Dataset: %s, num_instances: %d, num_features: %d, csr2csc finished. ###\n",
           param.path.c_str(),
           dataset.n_instances(),
           dataset.n_features());

    this->csr2csc(dataset);

    EXPECT_EQ(columns.n_column, this->n_column);
    EXPECT_EQ(columns.n_row, this->n_row);

    // --- test csc_val
    auto gpu_csc_val_data = columns.csc_val.host_data();
    auto cpu_csc_val_data = this->csc_val.host_data();
    for(int i = 0; i < columns.csc_val.size(); i++) {
        EXPECT_EQ(gpu_csc_val_data[i], cpu_csc_val_data[i]);
    }

    // --- test csc_row_idx
    auto gpu_csc_row_idx = columns.csc_row_idx.host_data();
    auto cpu_csc_row_idx = this->csc_row_idx.host_data();
    for(int i = 0; i < columns.csc_row_idx.size(); i++) {
        EXPECT_EQ(gpu_csc_row_idx[i], cpu_csc_row_idx[i]);
    }

    // --- test csc_col_ptr
    auto gpu_csc_col_ptr = columns.csc_col_ptr.host_data();
    auto cpu_csc_col_ptr = this->csc_col_ptr.host_data();
    for(int i = 0; i < columns.csc_col_ptr.size(); i++) {
        EXPECT_EQ(gpu_csc_col_ptr[i], cpu_csc_col_ptr[i]);
    }
}
