//
// Created by zeyi on 1/10/19.
//

#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "thundergbm/parser.h"

void Parser::parse_param(GBMParam &model_param, int argc, char **argv){
    model_param.depth = 6;
    model_param.n_trees = 40;
    model_param.n_device = 1;
    model_param.min_child_weight = 1;
    model_param.lambda = 1;
    model_param.gamma = 1;
    model_param.rt_eps = 1e-6;
    model_param.max_num_bin = 255;
    model_param.verbose = false;
    model_param.profiling = false;
    model_param.column_sampling_rate = 1;
    model_param.bagging = false;
    model_param.n_parallel_trees = 1;
    model_param.learning_rate = 1;
    model_param.objective = "reg:linear";
    model_param.num_class = 1;
    model_param.path = "../dataset/test_dataset.txt";
    model_param.out_model_name = "tgbm.model";
    model_param.in_model_name =  "tgbm.model";
    model_param.tree_method = "auto";

    if (argc < 2) {
        printf("Usage: <config>\n");
        exit(0);
    }

    //parsing parameter values from configuration file or command line
    auto parse_value = [&](const char *name_val){
        char name[256], val[256];
        if (sscanf(name_val, "%[^=]=%s", name, val) == 2) {
            string str_name(name);
            if(str_name.compare("max_depth") == 0)
                model_param.depth = atoi(val);
            else if(str_name.compare("num_round") == 0)
                model_param.n_trees = atoi(val);
            else if(str_name.compare("n_gpus") == 0)
                model_param.n_device = atoi(val);
            else if(str_name.compare("verbosity") == 0)
                model_param.verbose = atoi(val);
            else if(str_name.compare("profiling") == 0)
                model_param.profiling = atoi(val);
            else if(str_name.compare("data") == 0)
                model_param.path = val;
            else if(str_name.compare("max_bin") == 0)
                model_param.max_num_bin = atoi(val);
            else if(str_name.compare("colsample") == 0)
                model_param.column_sampling_rate = atof(val);
            else if(str_name.compare("bagging") == 0)
                model_param.bagging = atoi(val);
            else if(str_name.compare("num_parallel_tree") == 0)
                model_param.n_parallel_trees = atoi(val);
            else if(str_name.compare("eta") == 0 || str_name.compare("learning_rate") == 0)
                model_param.learning_rate = atof(val);
            else if(str_name.compare("objective") == 0)
                model_param.objective = val;
            else if(str_name.compare("num_class") == 0)
                model_param.num_class = atoi(val);
            else if(str_name.compare("min_child_weight") == 0)
                model_param.min_child_weight = atoi(val);
            else if(str_name.compare("lambda") == 0 || str_name.compare("reg_lambda") == 0)
                model_param.lambda = atof(val);
            else if(str_name.compare("gamma") == 0 || str_name.compare("min_split_loss") == 0)
                model_param.gamma = atof(val);
            else if(str_name.compare("model_out") == 0)
                model_param.out_model_name = val;
            else if(str_name.compare("model_in") == 0)
                model_param.in_model_name = val;
            else if(str_name.compare("tree_method") == 0)
                model_param.tree_method = val;
            else
                LOG(INFO) << "\"" << name << "\" is unknown option!";
        }
    };

    //read configuration file
    std::ifstream conf_file(argv[1]);
    std::string line;
    while (std::getline(conf_file, line))
    {
        //LOG(INFO) << line;
        parse_value(line.c_str());
    }

    //TODO: confirm handling spaces around "="
    for (int i = 0; i < argc; ++i) {
        parse_value(argv[i]);
    }//end parsing parameters
}

void Parser::load_model(GBMParam &model_param, vector<vector<Tree>> &boosted_model) {
    std::ifstream ifs(model_param.in_model_name);
    CHECK_EQ(ifs.is_open(), true);
    boost::archive::text_iarchive ia(ifs);
    ia & model_param.objective;
    ia & model_param.learning_rate;
    ia & model_param.num_class;
    //ia & model_param;
    ia & boosted_model;
    ifs.close();
}