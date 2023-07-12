//
// Created by zeyi on 1/9/19.
//
#include <fstream>
#include "cuda_runtime_api.h"

#include <thundergbm/tree.h>
#include <thundergbm/trainer.h>
#include <thundergbm/metric/metric.h>
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/reduce.h"
#include "time.h"
#include "thundergbm/booster.h"
#include "chrono"
#include <thundergbm/parser.h>
using namespace std;
long long total_sp_time = 0;
long long total_sort_time_hist = 0;
long long total_exact_prefix_sum_time = 0;

long long total_update_index1 = 0;
long long total_update_index2 = 0;

float convert_time = 0;
vector<vector<Tree>> TreeTrainer::train(GBMParam &param, const DataSet &dataset) {
    if (param.tree_method == "auto")
        if (dataset.n_features() > 20000)
            param.tree_method = "exact";
        else
            param.tree_method = "hist";

    //correct the number of classes
    if(param.objective.find("multi:") != std::string::npos || param.objective.find("binary:") != std::string::npos) {
        int num_class = dataset.label.size();
        if (param.num_class != num_class) {
            LOG(INFO) << "updating number of classes from " << param.num_class << " to " << num_class;
            param.num_class = num_class;
        }
        if(param.num_class > 2)
            param.tree_per_rounds = param.num_class;
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
    LOG(INFO)<<"total  hist construction time is "<<total_sort_time_hist/1e6;
    LOG(INFO)<<"total  exact prefix sum time is "<<total_exact_prefix_sum_time/1e6;
    LOG(INFO)<<"    total map array time1 is "<<total_update_index1/1e6;
    LOG(INFO)<<"    total map array time2 is "<<total_update_index2/1e6;
    LOG(INFO)<<"total update instance to node index array time is "<<total_sp_time/1e6;
    auto stop = timer.now();
    std::chrono::duration<float> training_time = stop - start;
    LOG(INFO)<<"convert time = "<<convert_time;
    LOG(INFO) << "training time = " << training_time.count();
    LOG(INFO)<<"total time = "<<convert_time+training_time.count();

    std::atexit([]() {
        SyncMem::clear_cache();
    });
	// SyncMem::clear_cache();
	return boosted_model;
}
