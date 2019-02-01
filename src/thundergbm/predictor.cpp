//
// Created by zeyi on 1/12/19.
//

#include "thundergbm/predictor.h"
#include <thundergbm/metric/metric.h>

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

void Predictor::predict(GBMParam& model_param, vector<vector<Tree>> &boosted_model, DataSet &dataSet){
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
        for(int iter = 0; iter < boosted_model.size(); iter++) {
            int num_tree = boosted_model[iter].size();
            float_type ave_val = 0;//average predicted value
            for(int t = 0; t < num_tree; t++) {//one iteration may have multiple trees (e.g., boosted Random Forests)
                const Tree::TreeNode *node_data = boosted_model[iter][t].nodes.host_data();
                Tree::TreeNode curNode = node_data[0];
                int cur_nid = 0; //node id
                while (!curNode.is_leaf) {
                    int fid = curNode.split_feature_id;
                    cur_nid = get_next_child(curNode, ins[fid]);
                    curNode = node_data[cur_nid];
                }
                ave_val += node_data[cur_nid].base_weight;
            }
            ave_val = ave_val / num_tree;
            predict_val[i] += ave_val;
        }//end all tree prediction
    }
    //store the predicted values into sync array
    SyncArray<float_type> y_predict;
    y_predict.resize(predict_val.size());
    y_predict.copy_from(predict_val.data(), predict_val.size());

    //convert the aggregated values to labels, probabilities or ranking scores.
    std::unique_ptr<ObjectiveFunction> obj;
    obj.reset(ObjectiveFunction::create(model_param.objective));
    obj->configure(model_param, dataSet);
    obj->predict_transform(y_predict);

    //compute metric
    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(model_param, dataSet);
    LOG(INFO) << metric->get_name().c_str() << "=" << metric->get_score(y_predict);
}