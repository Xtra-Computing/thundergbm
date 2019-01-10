//
// Created by zeyi on 1/10/19.
//

#include "thundergbm/thundergbm.h"
#include "thundergbm/param.h"
#include <thundergbm/trainer.h>
#include "thundergbm/param_parser.h"
#include <thundergbm/syncmem.h>

inline std::string trim(std::string& str)
{
    str.erase(0, str.find_first_not_of(' '));       //prefixing spaces
    str.erase(str.find_last_not_of(' ')+1);         //surfixing spaces
    return str;
}

int main(int argc, char **argv) {
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
    el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
    el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");

    GBMParam model_param;
    ParamParser parser;
    parser.parse_param(model_param, argc, argv);
    float_type rmse;
    //load model

    LOG(INFO) << "input file name: " << model_param.in_model_name;
    std::ifstream model_file(model_param.in_model_name);
    std::string line;
    vector<Tree> trees;
    trees.resize(2);
    trees[0].nodes.resize(256);
    trees[1].nodes.resize(256);
    model_param.depth = 6;
    Tree::TreeNode *nodes;
    int tid = -1;
    while (std::getline(model_file, line))
    {
        int nid, fid, lch_id, rch_id, missing_id;
        float_type sp_value, weight;
        LOG(INFO) << line;
        line = trim(line);
        std::size_t found = line.find("leaf");
        if (found != std::string::npos){
            LOG(INFO) << "parsing leaf";
            sscanf(line.c_str(), "%d:leaf=%f", &nid, &weight);
            Tree::TreeNode &node = trees[tid].nodes.host_data()[nid];
            node.final_id = nid;
            node.base_weight = weight;
            node.is_valid = true;
            node.is_pruned = false;
            node.is_leaf = true;
        }
        else {
            int fill_val = sscanf(line.c_str(), "%d:[f%d<%f] yes=%d,no=%d,missing=%d", &nid, &fid, &sp_value, &lch_id,
                                  &rch_id, &missing_id);
            LOG(INFO) << fill_val;
            if (fill_val == 6) {
                LOG(INFO) << "parsing internal";
                Tree::TreeNode &node = trees[tid].nodes.host_data()[nid];
                node.final_id = nid;
                node.lch_index = lch_id;
                node.rch_index = rch_id;
                node.split_value = sp_value;
                node.split_feature_id = fid;
                node.default_right = (missing_id == lch_id ? false : true);
                node.is_valid = true;
                node.is_pruned = false;
                node.is_leaf = false;
            } else {
                //start a new tree
                //Tree tree;
                //trees.emplace_back(tree);
                //trees.back().nodes.resize(256);
                //nodes = trees.back().nodes.host_data();
                tid++;
            }
        }
    }
    TreeTrainer trainer;
    trainer.save_trees(model_param, trees);
}
