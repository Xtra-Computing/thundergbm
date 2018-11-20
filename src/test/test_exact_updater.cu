//
// Created by jiashuai on 18-1-18.
//
#include <thundergbm/tree.h>
#include <thundergbm/dataset.h>
#include <thundergbm/updater/exact_updater.h>
#include <thundergbm/updater/hist_updater.h>
#include "gtest/gtest.h"
//#include "mpi.h"

class UpdaterTest : public ::testing::Test {
public:

    GBMParam param;
    bool verbose = false;

    void SetUp() override {
//        MPI_Init(NULL, NULL);
        //common param
        param.depth = 6;
        param.n_trees = 40;
        param.min_child_weight = 1;
        param.lambda = 1;
        param.gamma = 1;
        param.rt_eps = 1e-6;
        param.do_exact = true;
        param.n_device = 1;
        verbose = true;
//        MPI_Comm_size(MPI_COMM_WORLD, &param.n_executor);


        if (!verbose) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        }
//        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
    }

    void TearDown() {
//        MPI_Finalize();
    }

    float_type train_exact(GBMParam &param) {
        DataSet dataSet;
        dataSet.load_from_file(param.path);
        int n_instances = dataSet.n_instances();
        InsStat stats;
        vector<Tree> trees;
        SparseColumns columns;
        columns.from_dataset(dataSet);
        trees.resize(param.n_trees);
        stats.resize(n_instances);
        stats.y.copy_from(dataSet.y.data(), n_instances);

        int n_devices = param.n_device;
//        int rank;
//        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//        LOG(INFO) << "rank = " << rank;
//        MPI_Barrier(MPI_COMM_WORLD);
        vector<std::shared_ptr<SparseColumns>> v_columns;
        v_columns.resize(n_devices);
        for (int i = 0; i < n_devices; i++)
            v_columns[i].reset(new SparseColumns());
//        SparseColumns local_columns;
//        columns.get_shards(rank, param.n_executor, local_columns);
//        local_columns.to_multi_devices(v_columns);
        columns.to_multi_devices(v_columns);
        ExactUpdater updater(param);
        int round = 0;
        float_type rmse = 0;
        SyncMem::clear_cache();
        {
            TIMED_SCOPE(timerObj, "construct tree");
            for (Tree &tree:trees) {
                stats.updateGH();
                updater.grow(tree, v_columns, stats);
                tree.prune_self(param.gamma);
                LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
                predict_in_training(stats, tree);
                //next round
                round++;
                rmse = compute_rmse(stats);
                LOG(INFO) << "rmse = " << rmse;
            }
        }
        return rmse;
    }

    float_type train_hist(GBMParam &param) {
        DataSet dataSet;
        dataSet.load_from_file(param.path);
        int n_instances = dataSet.n_instances();
        InsStat stats;
        vector<Tree> trees;
        SparseColumns columns;
        columns.from_dataset(dataSet);
        trees.resize(param.n_trees);
        stats.resize(n_instances);
        stats.y.copy_from(dataSet.y.data(), n_instances);

        int n_devices = param.n_device;
//        int rank;
//        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//        LOG(INFO) << "rank = " << rank;
//        MPI_Barrier(MPI_COMM_WORLD);
        vector<std::shared_ptr<SparseColumns>> v_columns;
        v_columns.resize(n_devices);
        for (int i = 0; i < n_devices; i++)
            v_columns[i].reset(new SparseColumns());
//        SparseColumns local_columns;
//        columns.get_shards(rank, param.n_executor, local_columns);
//        local_columns.to_multi_devices(v_columns);
        columns.to_multi_devices(v_columns);
        HistUpdater updater(param);
        int round = 0;
        float_type rmse = 0;
        SyncMem::clear_cache();
        stats.updateGH();
        updater.init_cut(v_columns, stats, n_instances);
        {
            TIMED_SCOPE(timerObj, "construct tree");
            for (Tree &tree:trees) {
                stats.updateGH();
                updater.grow(tree, v_columns, stats);
                tree.prune_self(param.gamma);
                LOG(DEBUG) << string_format("\nbooster[%d]", round) << tree.dump(param.depth);
                predict_in_training(stats, tree);
                //next round
                round++;
            }
        }
        rmse = compute_rmse(stats);
        LOG(INFO) << "rmse = " << rmse;
        return rmse;
    }

    float_type compute_rmse(const InsStat &stats) {
        float_type sum_error = 0;
        const float_type *y_data = stats.y.host_data();
        const float_type *y_predict_data = stats.y_predict.host_data();
        for (int i = 0; i < stats.n_instances; ++i) {
            float_type e = y_predict_data[i] - y_data[i];
            sum_error += e * e;
        }
        float_type rmse = sqrt(sum_error / stats.n_instances);
        return rmse;
    }

    void predict_in_training(InsStat &stats, const Tree &tree) {

        TIMED_SCOPE(timerObj, "predict");
        float_type *y_predict_data = stats.y_predict.device_data();
        const int *nid_data = stats.nid.device_data();
        const Tree::TreeNode *nodes_data = tree.nodes.device_data();
        device_loop(stats.n_instances, [=]__device__(int i) {
            int nid = nid_data[i];
            while (nid != -1 && (nodes_data[nid].is_pruned)) nid = nodes_data[nid].parent_index;
            y_predict_data[i] += nodes_data[nid].base_weight;
        });
    }
};

class PerformanceTest : public UpdaterTest {
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

TEST_F(UpdaterTest, abalone_hist) {
    param.path = DATASET_DIR "abalone";
    float_type rmse = train_hist(param);//1674 ms
}


TEST_F(PerformanceTest, covtype_40_trees) {
    param.path = DATASET_DIR "covtype";
    train_hist(param);
}

TEST_F(PerformanceTest, e2006_40_trees) {
    param.path = DATASET_DIR "E2006.train";
    train_exact(param);
}

TEST_F(PerformanceTest, log1p_40_trees) {
    param.path = DATASET_DIR "log1p.E2006.train";
    train_exact(param);
}

TEST_F(PerformanceTest, higgs_40_trees) {
    param.path = DATASET_DIR "HIGGS";
    train_hist(param);
}

TEST_F(PerformanceTest, news20_40_trees) {
    param.path = DATASET_DIR "news20.binary";
    train_exact(param);
}

TEST_F(PerformanceTest, abalone) {
    param.path = DATASET_DIR "abalone";
    train_exact(param);
}

TEST_F(PerformanceTest, real_sim_40_trees) {
    param.path = DATASET_DIR "real-sim";
    train_exact(param);
}

TEST_F(PerformanceTest, susy_40_trees) {
    param.n_trees = 1;
    param.path = DATASET_DIR "SUSY";
    train_hist(param);
}

TEST_F(PerformanceTest, ins_40_trees) {
    param.path = DATASET_DIR "ins.libsvm";
    train_exact(param);
}

TEST_F(PerformanceTest, iris) {
    param.n_trees = 2;
    param.path = DATASET_DIR "iris.scale";
    train_hist(param);
}

TEST_F(PerformanceTest, iris_exact) {
    param.n_trees = 2;
    param.path = DATASET_DIR "iris.scale";
    train_exact(param);
}

TEST_F(PerformanceTest, year) {
    param.n_trees = 2;
    param.path = DATASET_DIR "YearPredictionMSD.scale";
    train_hist(param);
}

TEST_F(PerformanceTest, airline) {
    param.path = DATASET_DIR "airline_14col.data";
    train_exact(param);
}
//TEST_F(UpdaterTest, test_abalone_hist) {
//    param.path = DATASET_DIR "abalone";
//    float_type rmse = train_hist(param);
//    EXPECT_NEAR(rmse, 0.904459, 1e-5);
//}
//
//TEST_F(UpdaterTest, test_news20_hist) {
//    param.path = DATASET_DIR "news20.scale";
//    float_type rmse = train_hist(param);
//    EXPECT_NEAR(rmse, 2.55908, 1e-5);
//}

