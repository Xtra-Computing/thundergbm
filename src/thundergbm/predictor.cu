//
// Created by zeyi on 1/12/19.
//

#include "thundergbm/predictor.h"
#include <thundergbm/metric/metric.h>
#include "thundergbm/util/device_lambda.cuh"
#include "thundergbm/objective/objective_function.h"


vector<float_type> Predictor::predict(const GBMParam &model_param, const vector<vector<Tree>> &boosted_model,
                                      const DataSet &dataSet) {
    SyncArray<float_type> y_predict;
    predict_raw(model_param, boosted_model, dataSet, y_predict);
    //convert the aggregated values to labels, probabilities or ranking scores.
    std::unique_ptr<ObjectiveFunction> obj;
    obj.reset(ObjectiveFunction::create(model_param.objective));
    obj->configure(model_param, dataSet);

    //compute metric
    std::unique_ptr<Metric> metric;
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(model_param, dataSet);
    LOG(INFO) << metric->get_name().c_str() << " = " << metric->get_score(y_predict);

    obj->predict_transform(y_predict);
    vector<float_type> y_pred_vec(y_predict.size());
    memcpy(y_pred_vec.data(), y_predict.host_data(), sizeof(float_type) * y_predict.size());
    return y_pred_vec;
}

void Predictor::predict_raw(const GBMParam &model_param, const vector<vector<Tree>> &boosted_model,
                            const DataSet &dataSet, SyncArray<float_type> &y_predict) {
    TIMED_SCOPE(timerObj, "predict");
    int n_instances = dataSet.n_instances();
    int n_features = dataSet.n_features();

    //the whole model to an array
    int num_iter = boosted_model.size();
    int num_class = boosted_model.front().size();
    int num_node = boosted_model[0][0].nodes.size();
    int total_num_node = num_iter * num_class * num_node;
    //TODO: reduce the output size for binary classification
    y_predict.resize(n_instances * num_class);

    SyncArray<Tree::TreeNode> model(total_num_node);
    auto model_data = model.host_data();
    int tree_cnt = 0;
    for (auto &vtree:boosted_model) {
        for (auto &t:vtree) {
            memcpy(model_data + num_node * tree_cnt, t.nodes.host_data(), sizeof(Tree::TreeNode) * num_node);
            tree_cnt++;
        }
    }

    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "init trees");
    //copy instances from to GPU
    SyncArray<int> csr_col_idx(dataSet.csr_col_idx.size());
    SyncArray<float_type> csr_val(dataSet.csr_val.size());
    SyncArray<int> csr_row_ptr(dataSet.csr_row_ptr.size());
    csr_col_idx.copy_from(dataSet.csr_col_idx.data(), dataSet.csr_col_idx.size());
    csr_val.copy_from(dataSet.csr_val.data(), dataSet.csr_val.size());
    csr_row_ptr.copy_from(dataSet.csr_row_ptr.data(), dataSet.csr_row_ptr.size());

    //do prediction
    auto model_device_data = model.device_data();
    auto predict_data = y_predict.device_data();
    auto csr_col_idx_data = csr_col_idx.device_data();
    auto csr_val_data = csr_val.device_data();
    auto csr_row_ptr_data = csr_row_ptr.device_data();
    auto lr = model_param.learning_rate;
    PERFORMANCE_CHECKPOINT_WITH_ID(timerObj, "copy data");
    cudaDeviceSynchronize();

    //predict BLOCK_SIZE instances in a block, 1 thread for 1 instance
    int BLOCK_SIZE = 128;
    //determine whether we can use shared memory
    size_t smem_size = n_features * BLOCK_SIZE * sizeof(float_type);
    int NUM_BLOCK = (n_instances - 1) / BLOCK_SIZE + 1;

    float_type base_score = model_param.base_score;
    if(num_class>1)
        base_score = 0;

    if (smem_size <= 48 * 1024L) {//48KB shared memory for P100
        LOG(INFO) << "use shared memory to predict";
        //use shared memory to store dense instances
        anonymous_kernel([=]__device__() {
            auto get_next_child = [&](Tree::TreeNode node, float_type feaValue) {
                return feaValue < node.split_value ? node.lch_index : node.rch_index;
            };
            extern __shared__ float_type dense_data[];
            int iid = blockIdx.x * blockDim.x + threadIdx.x;
            float_type *thread_ins = dense_data + threadIdx.x * n_features;
            for (int i = 0; i < n_features; ++i) {
                thread_ins[i] = INFINITY;//mark as missing;
            }
            __syncthreads();
            //init dense data
            if (iid < n_instances) { //prevent out of bounds
                int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
                float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
                int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
                for (int i = 0; i < row_len; ++i) {
                    thread_ins[col_idx[i]] = row_val[i];
                }
            }
            __syncthreads();
            if(iid >= n_instances){ //prevent out of bounds
                return;
            }
            for (int t = 0; t < num_class; t++) {
                double sum = base_score; 
                auto predict_data_class = predict_data + t * n_instances;
                for (int iter = 0; iter < num_iter; iter++) {
                    const Tree::TreeNode *node_data = model_device_data + iter * num_class * num_node + t * num_node;
                    Tree::TreeNode curNode = node_data[0];
                    int cur_nid = 0; //node id
                    while (!curNode.is_leaf) {
                        int fid = curNode.split_feature_id;
                        float_type fval = thread_ins[fid];
                        if (fval < INFINITY)
                            cur_nid = get_next_child(curNode, fval);
                        else if (curNode.default_right)
                            cur_nid = curNode.rch_index;
                        else
                            cur_nid = curNode.lch_index;
                        curNode = node_data[cur_nid];
                    }
                    sum += lr * node_data[cur_nid].base_weight;
                }
                predict_data_class[iid] += sum;
            }
        },n_instances * n_features, smem_size, NUM_BLOCK, BLOCK_SIZE);
    } else {
        //use sparse format and binary search
        device_loop(n_instances, [=]__device__(int iid) {
            auto get_next_child = [&](Tree::TreeNode node, float_type feaValue) {
                return feaValue < node.split_value ? node.lch_index : node.rch_index;
            };
            auto get_val = [&](const int *row_idx, const float_type *row_val, int row_len, int idx,
                               bool *is_missing) -> float_type {
                //binary search to get feature value
                const int *left = row_idx;
                const int *right = row_idx + row_len;

                while (left != right) {
                    const int *mid = left + (right - left) / 2;
                    if (*mid == idx) {
                        *is_missing = false;
                        return row_val[mid - row_idx];
                    }
                    if (*mid > idx)
                        right = mid;
                    else left = mid + 1;
                }
                *is_missing = true;
                return 0;
            };
            int *col_idx = csr_col_idx_data + csr_row_ptr_data[iid];
            float_type *row_val = csr_val_data + csr_row_ptr_data[iid];
            int row_len = csr_row_ptr_data[iid + 1] - csr_row_ptr_data[iid];
            for (int t = 0; t < num_class; t++) {
                auto predict_data_class = predict_data + t * n_instances;
                //this base score only support binary classification now
                float_type sum = base_score;
                for (int iter = 0; iter < num_iter; iter++) {
                    const Tree::TreeNode *node_data = model_device_data + iter * num_class * num_node + t * num_node;
                    Tree::TreeNode curNode = node_data[0];
                    int cur_nid = 0; //node id
                    while (!curNode.is_leaf) {
                        int fid = curNode.split_feature_id;
                        bool is_missing;
                        float_type fval = get_val(col_idx, row_val, row_len, fid, &is_missing);
                        if (!is_missing)
                            cur_nid = get_next_child(curNode, fval);
                        else if (curNode.default_right)
                            cur_nid = curNode.rch_index;
                        else
                            cur_nid = curNode.lch_index;
                        curNode = node_data[cur_nid];
                    }
                    sum += lr * node_data[cur_nid].base_weight;
                }
                predict_data_class[iid] += sum;
            }//end all tree prediction
        });
    }
}
