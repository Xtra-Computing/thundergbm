//
// Created by zeyi on 1/9/19.
//
#include <fstream>
#include "cuda_runtime_api.h"
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
using namespace std;

vector<vector<Tree>> TreeTrainer::train(GBMParam &param, const DataSet &dataset) {
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
    ofstream out_model_file(param.out_model_name, ios::binary);
    assert(out_model_file.is_open());
    int length = param.objective.length();
    out_model_file.write((char*)&length, sizeof(length));
    out_model_file.write(param.objective.c_str(), param.objective.length());
    out_model_file.write((char*)&param.learning_rate, sizeof(param.learning_rate));
    out_model_file.write((char*)&param.num_class, sizeof(param.num_class));
    int label_size = dataset.label.size();
    out_model_file.write((char*)&label_size, sizeof(label_size));
    out_model_file.write((char*)&dataset.label[0], dataset.label.size() * sizeof(float_type));
    int boosted_model_size = boosted_model.size();
    out_model_file.write((char*)&boosted_model_size, sizeof(boosted_model_size));
    for (int j = 0; j < boosted_model.size(); ++j) {
        int boosted_model_j_size = boosted_model[j].size();
        out_model_file.write((char*)&boosted_model_j_size, sizeof(boosted_model_j_size));
        for (int i = 0; i < boosted_model_j_size; ++i) {
            size_t syn_node_size = boosted_model[j][i].nodes.size();
            out_model_file.write((char*)&syn_node_size, sizeof(syn_node_size));
            out_model_file.write((char*)boosted_model[j][i].nodes.host_data(), syn_node_size * sizeof(Tree::TreeNode));
        }
    }
    out_model_file.close();
	SyncMem::clear_cache();
	return boosted_model;
}
