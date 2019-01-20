//
// Created by zeyi on 1/12/19.
//

#include "thundergbm/predictor.h"
#include "thundergbm/trainer.h"

int Predictor::get_next_child(const Tree::TreeNode &node, float_type feaValue)
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

void Predictor::predict(vector<Tree> &trees, DataSet &dataSet){
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
                cur_nid = get_next_child(curNode, ins[fid]);
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

//    float_type rmse = trainer.compute_rmse(stat);
//    printf("predicted rmse = %f\n", rmse);
}