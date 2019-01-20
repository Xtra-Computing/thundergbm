//
// Created by zeyi on 1/9/19.
//
#include <fstream>
#include <thundergbm/tree.h>
#include <thundergbm/trainer.h>
#include <thundergbm/metric/metric.h>
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/reduce.h"
#include "time.h"
#include "thundergbm/booster.h"

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

void TreeTrainer::train(GBMParam &param) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    if (param.tree_method == "auto")
        if (dataset.n_features() > 20000)
            param.tree_method = "exact";
        else
            param.tree_method = "hist";

    vector<vector<Tree>> boosted_model;
    Booster booster;
    booster.init(dataset, param);
    clock_t start = clock();
    for (int i = 0; i < param.n_trees; ++i) {
        booster.boost(boosted_model);
    }
    clock_t stop = clock();
    LOG(INFO) << "training time = " << (double) (stop - start) / CLOCKS_PER_SEC;
}
