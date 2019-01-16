//
// Created by zeyi on 1/9/19.
//
#include <fstream>
#include <thundergbm/tree.h>
#include <thundergbm/updater/exact_updater.h>
#include <thundergbm/updater/hist_updater.h>
#include <thundergbm/trainer.h>
#include <thundergbm/metric/metric.h>
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/reduce.h"

float_type TreeTrainer::compute_rmse(const InsStat &stats) {
    TIMED_FUNC(timerObj);
    SyncArray<float_type> sq_err(stats.n_instances);
    auto sq_err_data = sq_err.device_data();
    const float_type *y_data = stats.y.device_data();
    const float_type *y_predict_data = stats.y_predict.device_data();
    device_loop(stats.n_instances, [=] __device__(int i) {
        float_type e = y_predict_data[i] - y_data[i];
        sq_err_data[i] = e * e;
    });
    float_type rmse =
            sqrt(thrust::reduce(thrust::cuda::par, sq_err.device_data(), sq_err.device_end()) / stats.n_instances);
    return rmse;
}

void TreeTrainer::save_trees(GBMParam &param, vector<Tree> &trees) {
    std::ofstream out(param.out_model_name);
    int round = 0;
    for (Tree &tree:trees) {
        string str_tree = string_format("booster[%d]:", round) + tree.dump(param.depth);
        //LOG(INFO) << "\n" << str_tree;
        out << str_tree;
        round++;
    }
    out.close();
}

float_type TreeTrainer::train(GBMParam &param) {
    dataSet.load_from_file(param.path, param);
    float_type rmse;
    if (param.tree_method.compare("exact") == 0)
        rmse = train_exact(param);
    else if (param.tree_method.compare("hist") == 0)
        rmse = train_hist(param);
    else {
        bool exact_sp_producer = false;
        if (dataSet.n_features() > 20000)//#TODO: use data set density ratio
            exact_sp_producer = true;
        if (exact_sp_producer == true)
            rmse = train_exact(param);
        else
            rmse = train_hist(param);
    }
    return rmse;
}

float_type TreeTrainer::train_exact(GBMParam &param) {
    LOG(INFO) << "using exact split to train the trees";
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
            //next round
            round++;
            rmse = compute_rmse(updater.shards.front()->stats);
            LOG(INFO) << "rmse = " << rmse;
        }
        save_trees(param, trees);
    }
    return rmse;
}

///// upgrading
//float_type TreeTrainer::train_exact(GBMParam &param) {
//    DataSet dataSet;
//    dataSet.load_from_file(param.path, param);
//    int n_instances = dataSet.n_instances();
//    vector<Tree> trees;
//    trees.resize(param.n_trees);
//
//    ExactUpdater updater(param);
//    updater.init(dataSet);
//    int round = 0;
//    float_type rmse = 0;
//    SyncMem::clear_cache();
//    {
//        TIMED_SCOPE(timerObj, "construct tree");
//        for (Tree &tree:trees) {
//            updater.grow(tree);
//            //next round
//            round++;
//            rmse = compute_rmse(updater.shards.front()->stats);
//            LOG(INFO) << "rmse = " << rmse;
//        }
//        save_trees(param, trees);
//    }
//    return rmse;
//}

float_type TreeTrainer::train_hist(GBMParam &param) {
    LOG(INFO) << "using histogram based approach to find split";
    SyncMem::clear_cache();

    vector<vector<Tree>> trees;
    vector<HistUpdater::ShardT> shards(param.n_device);

    //TODO refactor these
    SparseColumns columns;
    columns.from_dataset(dataSet);
    vector<std::unique_ptr<SparseColumns>> v_columns(param.n_device);
    for (int i = 0; i < param.n_device; ++i) {
        v_columns[i].reset(&shards[i].columns);
    }
    columns.to_multi_devices(v_columns);

    HistUpdater updater(param);
    HistUpdater::for_each_shard(shards, [&](Shard &shard) {
        int n_instances = shard.columns.n_row;
        shard.stats.resize(n_instances);
        shard.stats.y.copy_from(dataSet.y.data(), n_instances);
        shard.stats.obj.reset(ObjectiveFunction::create(param.objective));
        shard.stats.obj->configure(param, dataSet);
        shard.param = param;
        shard.param.learning_rate /= param.n_parallel_trees;//average trees in one iteration
    });
    updater.init(shards);

    SyncMem::clear_cache();

    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(shards.front().stats.obj->default_metric()));
    metric->configure(param, dataSet);

    int round = 0;
    float_type score = 0;
    {
        TIMED_SCOPE(timerObj, "construct tree");
        int n_instances = shards.front().stats.n_instances;
        SyncArray<GHPair> all_gh_pair(n_instances * param.num_class);
        SyncArray<float_type> all_y(n_instances * param.num_class);
        for (int iter = 0; iter < param.n_trees; iter++) {
            //one boosting iteration

            trees.emplace_back();
            vector<Tree> &tree = trees.back();
            tree.resize(param.n_parallel_trees);
            if (param.num_class == 1) {
                //update gradient
                HistUpdater::for_each_shard(shards, [&](Shard &shard) {
                    shard.stats.update_gradient();
                    LOG(DEBUG) << "gh = " << shard.stats.gh_pair;
                    if (updater.param.bagging) {
                        shard.stats.gh_pair_backup.resize(shard.stats.n_instances);
                        shard.stats.gh_pair_backup.copy_from(shard.stats.gh_pair);
                    }
                });
                updater.grow(tree, shards);

                //next round
                round++;
                score = metric->get_score(shards.front().stats.y_predict);
            } else {
                shards.front().stats.obj->get_gradient(shards.front().stats.y, all_y, all_gh_pair);
                for (int i = 0; i < param.num_class; ++i) {
                    trees.emplace_back();
                    vector<Tree> &tree = trees.back();
                    tree.resize(param.n_parallel_trees);
                    HistUpdater::for_each_shard(shards, [&](Shard &shard) {
                        shard.stats.gh_pair.copy_from(all_gh_pair.device_data() + i * n_instances, n_instances);
                        shard.stats.y_predict.copy_from(all_y.device_data() + i * n_instances, n_instances);
                    });
                    updater.grow(tree, shards);
                    CUDA_CHECK(cudaMemcpy(all_y.device_data() + i * n_instances,
                                          shards.front().stats.y_predict.device_data(),
                                          sizeof(float_type) * n_instances, cudaMemcpyDefault));
                }
                score = metric->get_score(all_y);
            }
            LOG(INFO) << metric->get_name() << " = " << score;
        }
//        LOG(INFO) << trees.back().back().dump(param.depth);
    }
    for (int i = 0; i < param.n_device; ++i) {
        v_columns[i].release();
    }
    return score;
}