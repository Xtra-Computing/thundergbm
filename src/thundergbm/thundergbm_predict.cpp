//
// Created by zeyi on 1/10/19.
//

#include "thundergbm/thundergbm.h"
#include "thundergbm/param.h"
#include <thundergbm/trainer.h>
#include "thundergbm/param_parser.h"
#include <thundergbm/syncmem.h>
#include <thundergbm/dataset.h>

void load_model(GBMParam &model_param, vector<Tree> &trees){
    std::ifstream model_file(model_param.in_model_name);
    std::string line;
    trees.resize(2);
    trees[0].nodes.resize(512);
    trees[1].nodes.resize(512);

    model_param.depth = 6;
    Tree::TreeNode *nodes;
    int tid = -1;
    while (std::getline(model_file, line))
    {
        int nid, fid, lch_id, rch_id, missing_id;
        float_type sp_value, weight;
        //LOG(INFO) << line;
        std::size_t found = line.find("leaf");
        if (found != std::string::npos){
            //LOG(INFO) << "parsing leaf";
            sscanf(line.c_str(), "%d:leaf=%f", &nid, &weight);
            Tree::TreeNode &node = trees[tid].nodes.host_data()[nid];
            node.final_id = nid;
            node.base_weight = weight;
            node.is_valid = true;
            node.is_pruned = false;
            node.is_leaf = true;
            node.lch_index = -1;
            node.rch_index = -1;
        }
        else {
            int fill_val = sscanf(line.c_str(), "%d:[f%d<%f] yes=%d,no=%d,missing=%d", &nid, &fid, &sp_value, &lch_id,
                                  &rch_id, &missing_id);
            //LOG(INFO) << fill_val;
            if (fill_val == 6) {
               // LOG(INFO) << "parsing internal";
                Tree::TreeNode &node = trees[tid].nodes.host_data()[nid];
                node.final_id = nid;
                node.lch_index = lch_id;
                node.rch_index = rch_id;
                node.split_value = sp_value;
                node.split_feature_id = fid - 1;//keep consistent with save tree
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
}

int GetNext(const Tree::TreeNode &node, float_type feaValue)
{
    if(feaValue == 0){//this is a missing value
        if(node.default_right == false)
            return node.lch_index;
        else
            return node.rch_index;
    }

    if(feaValue < node.split_value)
        return node.lch_index;
    else
        return node.rch_index;
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
    //load model
    vector<Tree> trees;
    load_model(model_param, trees);

    DataSet dataSet;
    dataSet.load_from_file(model_param.path);
    int n_instances = dataSet.n_instances();
    int n_feature = dataSet.n_features();
    vector<float_type> predict_val(n_instances);

#pragma omp parallel for
    for(int i = 0; i < n_instances; i++){
        //fill dense
        vector<float_type> ins(n_feature, 0);
        int row_start_pos = dataSet.csr_row_ptr[i];
        int row_len = dataSet.csr_val.size() - row_start_pos;
        if(i < n_instances - 1)
            row_len = dataSet.csr_row_ptr[i + 1] - row_start_pos;
        for(int fv = row_start_pos; fv < row_start_pos + row_len; fv++){
            int fid = dataSet.csr_col_idx[fv];
            ins[fid] = dataSet.csr_val[fv];
        }

        //predict
        for(int t = 0; t < trees.size(); t++) {
            const Tree::TreeNode *node_data = trees[t].nodes.host_data();
            Tree::TreeNode curNode = node_data[0];
            int cur_nid = 0; //node id
            while (!curNode.is_leaf)
            {
                int fid = curNode.split_feature_id;
                cur_nid = GetNext(curNode, ins[fid]);
                curNode = node_data[cur_nid];
            }
            predict_val[i] += node_data[cur_nid].base_weight;
        }//end all tree prediction
    }
    //compute rmse
    LOG(INFO) << "compute RMSE";
    TreeTrainer trainer;
    InsStat stat(n_instances);
    stat.y.resize(dataSet.y.size());
    stat.y.copy_from(dataSet.y.data(), dataSet.y.size());
    stat.y_predict.resize(predict_val.size());
    stat.y_predict.copy_from(predict_val.data(), predict_val.size());

    float_type rmse = trainer.compute_rmse(stat);
    LOG(INFO) << "rmse = " << rmse;
}
