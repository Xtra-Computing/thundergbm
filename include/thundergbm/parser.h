//
// Created by zeyi on 1/10/19.
//

#ifndef THUNDERGBM_PARAM_PARSER_H
#define THUNDERGBM_PARAM_PARSER_H

#include "tree.h"
#include "dataset.h"

class Parser{
public:
    void parse_param(GBMParam &model_param, int argc, char **argv);
    void load_model(string model_path, GBMParam &model_param, vector<vector<Tree>> &boosted_model, DataSet &dataSet);
    void save_model(string model_path, GBMParam &model_param, vector<vector<Tree>> &boosted_model, DataSet &dataSet);
};

#endif //THUNDERGBM_PARAM_PARSER_H
