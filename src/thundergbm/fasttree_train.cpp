//
// Created by zeyi on 1/9/19.
//


#include "thundergbm/thundergbm.h"
#include "thundergbm/param.h"
#include <thundergbm/trainer.h>

void parse_param(GBMParam &tree_param, int argc, char **argv){
    tree_param.depth = 6;
    tree_param.n_trees = 40;
    tree_param.n_device = 1;
    tree_param.min_child_weight = 1;
    tree_param.lambda = 1;
    tree_param.gamma = 1;
    tree_param.rt_eps = 1e-6;
    tree_param.max_num_bin = 255;
    tree_param.verbose = false;
    tree_param.column_sampling_rate = 1;
    tree_param.bagging = false;
    tree_param.n_parallel_trees = 1;
    tree_param.learning_rate = 1;
    tree_param.objective = "reg:linear";
    tree_param.num_class = 1;

    //TODO: confirm handling spaces around "="
    for (int i = 0; i < argc; ++i) {
        char name[256], val[256];
        if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
            string str_name(name);
            if(str_name.compare("max_depth") == 0)
                tree_param.depth = atoi(val);
            else if(str_name.compare("num_round") == 0)
                tree_param.n_trees = atoi(val);
            else if(str_name.compare("n_gpus") == 0)
                tree_param.n_device = atoi(val);
            else if(str_name.compare("verbosity") == 0)
                tree_param.verbose = atoi(val);
            else if(str_name.compare("data") == 0)
                tree_param.path = val;
            else if(str_name.compare("max_bin") == 0)
                tree_param.max_num_bin = atoi(val);
            else if(str_name.compare("colsample") == 0)
                tree_param.column_sampling_rate = atof(val);
            else if(str_name.compare("bagging") == 0)
                tree_param.bagging = atoi(val);
            else if(str_name.compare("num_parallel_tree") == 0)
                tree_param.n_parallel_trees = atoi(val);
            else if(str_name.compare("eta") == 0 || str_name.compare("learning_rate") == 0)
                tree_param.learning_rate = atof(val);
            else if(str_name.compare("objective") == 0)
                tree_param.objective = val;
            else if(str_name.compare("num_class") == 0)
                tree_param.num_class = atoi(val);
            else if(str_name.compare("min_child_weight") == 0)
                tree_param.min_child_weight = atoi(val);
            else if(str_name.compare("lambda") == 0 || str_name.compare("reg_lambda") == 0)
                tree_param.lambda = atof(val);
            else if(str_name.compare("gamma") == 0 || str_name.compare("min_split_loss") == 0)
                tree_param.gamma = atof(val);
            else
                LOG(INFO) << "\"" << name << "\" is unknown option!";
        }
    }//end parsing parameters
}

int main(int argc, char **argv) {
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

    GBMParam tree_param;
    parse_param(tree_param, argc, argv);
    float_type rmse;
    TreeTrainer trainer;
    bool exact_sp_producer = true;
    if(exact_sp_producer)
        rmse = trainer.train_exact(tree_param);
    else
        rmse = trainer.train_hist(tree_param);
    LOG(INFO) << "RMSE is " << rmse;
}