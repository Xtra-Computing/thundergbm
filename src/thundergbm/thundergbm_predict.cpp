//
// Created by zeyi on 1/10/19.
//

#include "thundergbm/thundergbm.h"
#include "thundergbm/param.h"
#include <thundergbm/trainer.h>
#include "thundergbm/parser.h"
#include <thundergbm/syncmem.h>
#include <thundergbm/dataset.h>

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
    Parser parser;
    parser.parse_param(model_param, argc, argv);
    //load model
    vector<Tree> trees;
    parser.load_model(model_param, trees);

    DataSet dataSet;
    dataSet.load_from_file(model_param.path, model_param);
    int n_instances = dataSet.n_instances();
    int n_feature = dataSet.n_features();
    vector<float_type> predict_val(n_instances);

//#pragma omp parallel for num_threads(10)
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
        //LOG(INFO) << ".";
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
