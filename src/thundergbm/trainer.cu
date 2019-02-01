//
// Created by zeyi on 1/9/19.
//
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <thundergbm/tree.h>
#include <thundergbm/trainer.h>
#include <thundergbm/metric/metric.h>
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/reduce.h"
#include "time.h"
#include "thundergbm/booster.h"
#include "chrono"

void TreeTrainer::dump_model(GBMParam &param, vector<Tree> &trees) {
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

vector<vector<Tree>> TreeTrainer::train(GBMParam &param) {
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    if (param.tree_method == "auto")
        if (dataset.n_features() > 20000)
            param.tree_method = "exact";
        else
            param.tree_method = "hist";

    //correct the number of classes
    if(param.objective.find("multi:") != std::string::npos) {
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
    }
    else if(param.objective.find("reg:") != std::string::npos){
        param.num_class = 1;
    }

    vector<vector<Tree>> boosted_model;
    Booster booster;
    booster.init(dataset, param);
    std::chrono::high_resolution_clock timer;
    auto start = timer.now();
    for (int i = 0; i < param.n_trees; ++i) {
        //one iteration may produce multiple trees, depending on objectives
        booster.boost(boosted_model);
    }
    auto stop = timer.now();
    std::chrono::duration<float> training_time = stop - start;
    LOG(INFO) << "training time = " << training_time.count();

    //save model
    std::ofstream ofs(param.out_model_name);
    boost::archive::text_oarchive oa(ofs);
    oa & param.objective;
    oa & boosted_model;
    ofs.close();

    return boosted_model;
}
