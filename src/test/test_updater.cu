//
// Created by jiashuai on 18-1-18.
//
#include <thundergbm/tree.h>
#include <thundergbm/dataset.h>
#include <thundergbm/updater/exact_updater.h>
#include <thundergbm/updater/hist_updater.h>
#include <thundergbm/syncmem.h>
#include "gtest/gtest.h"
#include "cuda_profiler_api.h"
//#include "mpi.h"

extern GBMParam global_test_param;
class UpdaterTest : public ::testing::Test {
public:

    GBMParam param = global_test_param;

    void SetUp() override {


        if (!param.verbose) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        }
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
    }

    void TearDown() {
//        MPI_Finalize();
        SyncMem::clear_cache();
    }

    float_type train_exact(GBMParam &param) {
        DataSet dataSet;
        dataSet.load_from_file(param.path);
        int n_instances = dataSet.n_instances();
        vector<Tree> trees;
        trees.resize(param.n_trees);

        ExactUpdater updater(param);
        updater.init(dataSet);
        int round = 0;
        float_type rmse = 0;
        SyncMem::clear_cache();
        {
            TIMED_SCOPE(timerObj, "construct tree");
            for (Tree &tree:trees) {
                updater.grow(tree);
                LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
                //next round
                round++;
                rmse = compute_rmse(updater.shards.front()->stats);
                LOG(INFO) << "rmse = " << rmse;
            }
        }
        return rmse;
    }

    float_type train_hist(GBMParam &param) {
        DataSet dataSet;
        dataSet.load_from_file(param.path);

        int round = 0;
        float_type rmse = 0;
        SyncMem::clear_cache();
        vector<vector<Tree>> trees(param.n_trees);
        HistUpdater updater(param);

        updater.init(dataSet);
        SyncMem::clear_cache();
        {
            TIMED_SCOPE(timerObj, "construct tree");
            for (vector<Tree> &tree:trees) {
                tree.resize(param.n_parallel_trees);
                updater.grow(tree);
//                LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
                //next round
                round++;
                rmse = compute_rmse(updater.shards.front()->stats);
                LOG(INFO) << "rmse = " << rmse;
            }
        }
        return rmse;
    }

    float_type compute_rmse(const InsStat &stats) {
        TIMED_FUNC(timerObj);
        SyncArray<float_type> sq_err(stats.n_instances);
        auto sq_err_data = sq_err.device_data();
        const float_type *y_data = stats.y.device_data();
        const float_type *y_predict_data = stats.y_predict.device_data();
        device_loop(stats.n_instances, [=]__device__(int i) {
            float_type e = y_predict_data[i] - y_data[i];
            sq_err_data[i] = e * e;
        });
        float_type rmse =
                sqrt(thrust::reduce(thrust::cuda::par, sq_err.device_data(), sq_err.device_end()) / stats.n_instances);
        return rmse;
    }

};

class Exact : public UpdaterTest {
};
class Hist : public UpdaterTest {
};

TEST_F(UpdaterTest, news20_40_trees_same_as_xgboost) {
    param.path = DATASET_DIR "news20.scale";
    float_type rmse = train_exact(param);//5375 ms
    EXPECT_NEAR(rmse, 2.55275, 1e-5);
}

TEST_F(UpdaterTest, abalone_40_trees_same_as_xgboost) {
    param.path = DATASET_DIR "abalone";
    float_type rmse = train_exact(param);//1674 ms
    EXPECT_NEAR(rmse, 0.803684, 1e-5);
}

TEST_F(UpdaterTest, iris) {
    param.n_trees = 2;
    param.path = DATASET_DIR "iris.scale";
    train_hist(param);
}

TEST_F(UpdaterTest, iris_exact) {
    param.n_trees = 2;
    param.path = DATASET_DIR "iris.scale";
    train_exact(param);
}

TEST_F(Exact, covtype) {
    param.path = DATASET_DIR "covtype";
    train_exact(param);
}

TEST_F(Exact, e2006) {
    param.path = DATASET_DIR "E2006.train";
    train_exact(param);
}

TEST_F(Exact, higgs) {
    param.path = DATASET_DIR "HIGGS";
    train_exact(param);
}

TEST_F(Exact, ins) {
    param.path = DATASET_DIR "ins.libsvm";
    train_exact(param);
}

TEST_F(Exact, log1p) {
    param.path = DATASET_DIR "log1p.E2006.train";
    train_exact(param);
}


TEST_F(Exact, news20) {
    param.path = DATASET_DIR "news20.binary";
    train_exact(param);
}

TEST_F(Exact, real_sim) {
    param.path = DATASET_DIR "real-sim";
    train_exact(param);
}

TEST_F(Exact, susy) {
    param.path = DATASET_DIR "SUSY";
    train_exact(param);
}

TEST_F(Hist, covtype) {
    param.path = DATASET_DIR "covtype";
    train_hist(param);
}

TEST_F(Hist, higgs) {
    param.path = DATASET_DIR "HIGGS";
    train_hist(param);
}

TEST_F(Hist, ins) {
    param.path = DATASET_DIR "ins.libsvm";
    train_hist(param);
}

TEST_F(Hist, susy) {
    param.path = DATASET_DIR "SUSY";
    train_hist(param);
}

TEST_F(Hist, any) {
    train_hist(param);
}
