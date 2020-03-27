//
// Created by zeyi on 1/12/19.
//

#include <thundergbm/trainer.h>
#include "thundergbm/parser.h"
#include <thundergbm/dataset.h>
#include "thundergbm/predictor.h"
#include "thundergbm/objective/objective_function.h"
#include <thundergbm/metric/metric.h>

extern "C" {
    void set_logger(int verbose) {
        if(verbose == 0) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
        }
        else if (verbose == 1) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        }
    }


    void sparse_train_scikit(int row_size, float *val, int *row_ptr, int *col_ptr, float *label,
                             int depth, int n_trees, int n_device, float min_child_weight, float lambda,
                             float gamma, int max_num_bin, int verbose, float column_sampling_rate,
                             int bagging, int n_parallel_trees, float learning_rate, char *obj_type,
                             int *num_class, char *tree_method, Tree *&model, int *tree_per_iter, float *group_label,
                             int *group, int num_group=0) {
//        train_succeed[0] = 1;
        //init model_param
        GBMParam model_param;
        model_param.depth = depth;
        model_param.n_trees = n_trees;
        model_param.n_device = n_device;
        model_param.min_child_weight = min_child_weight;
        model_param.lambda = lambda;
        model_param.gamma = gamma;
        model_param.max_num_bin = max_num_bin;
        model_param.verbose = verbose;
        model_param.column_sampling_rate = column_sampling_rate;
        model_param.bagging = bagging;
        model_param.n_parallel_trees = n_parallel_trees;
        model_param.learning_rate = learning_rate;
        model_param.objective = obj_type;
        model_param.num_class = *num_class;
        model_param.tree_method = tree_method;
        model_param.rt_eps = 1e-6;
        model_param.tree_per_rounds = 1;

        set_logger(verbose);

        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        DataSet train_dataset;
        train_dataset.load_from_sparse(row_size, val, row_ptr, col_ptr, label, group, num_group, model_param);
        *num_class = model_param.num_class;
        TreeTrainer trainer;
        vector<vector<Tree> > boosted_model = trainer.train(model_param, train_dataset);
        *tree_per_iter =  (int)(boosted_model[0].size());
        //TODO: track potential memory leakage
        model = new Tree[n_trees * (*tree_per_iter)];
		CHECK_EQ(n_trees, boosted_model.size()) << n_trees << " v.s. " << boosted_model.size();
		for(int i = 0; i < n_trees; i++)
		{
			for(int j = 0; j < *tree_per_iter; j++){
                model[i * (*tree_per_iter) + j] = boosted_model[i][j];
			}
		}
        for (int i = 0; i < train_dataset.label.size(); ++i) {
            group_label[i] = train_dataset.label[i];
        }
        // SyncMem::clear_cache();
        int gpu_num;
        cudaError_t err = cudaGetDeviceCount(&gpu_num);
        std::atexit([](){
            SyncMem::clear_cache();
        });
    }//end sparse_model_scikit


    void sparse_predict_scikit(int row_size, float *val, int *row_ptr, int *col_ptr, float *y_pred, Tree *&model,
            int n_trees, int trees_per_iter, char *objective, int num_class, float learning_rate, float *group_label,
            int *group, int num_group=0, int verbose=1){
        //load model
        GBMParam model_param;
        model_param.objective = objective;
        model_param.learning_rate = learning_rate;
        model_param.num_class = num_class;
        DataSet dataSet;
        dataSet.load_from_sparse(row_size, val, row_ptr, col_ptr, NULL, group, num_group, model_param);
        set_logger(verbose);
        dataSet.label.clear();
        for (int i = 0; i < num_class; ++i) {
            dataSet.label.emplace_back(group_label[i]);
        }
        //TODO: get ride of reading param and model
        //predict
        Predictor pred;
        SyncArray<float_type> y_predict;
		vector<vector<Tree>> boosted_model_in_mem;
        for(int i = 0; i < n_trees; i++){
			boosted_model_in_mem.push_back(vector<Tree>());
			CHECK_NE(model, NULL) << "model is null!";
			for(int j = 0; j < trees_per_iter; j++) {
                boosted_model_in_mem[i].push_back(model[i * trees_per_iter + j]);
            }
		}
        pred.predict_raw(model_param, boosted_model_in_mem, dataSet, y_predict);
        //convert the aggregated values to labels, probabilities or ranking scores.
        std::unique_ptr<ObjectiveFunction> obj;
        obj.reset(ObjectiveFunction::create(model_param.objective));
        obj->configure(model_param, dataSet);
        obj->predict_transform(y_predict);
        vector<float_type> y_pred_vec(y_predict.size());
        memcpy(y_pred_vec.data(), y_predict.host_data(), sizeof(float_type) * y_predict.size());
        for(int i = 0; i < y_pred_vec.size(); i++){
            y_pred[i] = y_pred_vec[i];
        }//float_type may be double/float, so convert to float for both cases.
    }

    void save(char *model_path, char *objective, float learning_rate, int num_class, int n_trees,
            int trees_per_iter, Tree *&model, float *group_label){
        GBMParam model_param;
        model_param.objective = objective;
        model_param.learning_rate = learning_rate;
        model_param.num_class = num_class;
        model_param.n_trees = n_trees;
        vector<vector<Tree>> boosted_model;
        for(int i = 0; i < n_trees; i++){
            boosted_model.push_back(vector<Tree>());
            CHECK_NE(model, NULL) << "model is null!";
            for(int j = 0; j < trees_per_iter; j++) {
                boosted_model[i].push_back(model[i * trees_per_iter + j]);
            }
        }
        Parser parser;
        DataSet dataset;
        dataset.label.clear();
        if(num_class != 1){
            for (int i = 0; i < num_class; ++i) {
                dataset.label.push_back(group_label[i]);
            }
        }
        parser.save_model(model_path, model_param, boosted_model, dataset);
    }
    void load_model(char *model_path, float *learning_rate, int *num_class, int *n_trees,
              int *trees_per_iter, Tree *&model){
        GBMParam model_param;
        vector<vector<Tree>> boosted_model;
        DataSet dataset;
        Parser parser;
        parser.load_model(model_path, model_param, boosted_model, dataset);
        *learning_rate = model_param.learning_rate;
        *num_class = model_param.num_class;
        *n_trees = model_param.n_trees;
        *trees_per_iter = (int)(boosted_model[0].size());
        model = new Tree[*n_trees * (*trees_per_iter)];
        //LOG(INFO) << *learning_rate;
        //LOG(INFO) << *num_class;
        //LOG(INFO) << *n_trees;
        //LOG(INFO) << *trees_per_iter;
        CHECK_EQ(*n_trees, boosted_model.size()) << n_trees << " v.s. " << boosted_model.size();
        for(int i = 0; i < *n_trees; i++)
        {
            for(int j = 0; j < *trees_per_iter; j++){
                model[i * (*trees_per_iter) + j] = boosted_model[i][j];
            }
        }

    }
    void load_config(char *model_path, float *group_label){
        DataSet dataset;
        GBMParam model_param;
        vector<vector<Tree>> boosted_model;
        Parser parser;
        parser.load_model(model_path, model_param, boosted_model, dataset);
        for (int i = 0; i < dataset.label.size(); ++i) {
            group_label[i] = dataset.label[i];
        }
    }

    void get_n_nodes(Tree* &model, int *n_nodes, int n_trees, int tree_per_iter){
        for(int i = 0; i < n_trees; i++){
            for(int j = 0; j < tree_per_iter; j++){
                n_nodes[i * tree_per_iter + j] = model[i * tree_per_iter + j].nodes.size();
//                std::cout<<n_nodes[i]<<" ";
//                if(model[i * tree_per_iter + j].final_n_nodes != 0) {
//                    n_nodes[i] = model[i * tree_per_iter + j].final_n_nodes;
//                    std::cout<<"final n nodes:"<<model[i * tree_per_iter + j].final_n_nodes;
//                }
//                else
//                    n_nodes[i] = model[i * tree_per_iter + j].nodes.size();
            }
        }
    }

    void get_a_tree(Tree* &model, int tree_id, int n_nodes, int* children_left, int* children_right, int* children_default,
            int* features, float* thresholds, float* values, float* node_sample_weights){
        Tree& tree = model[tree_id];
        CHECK(n_nodes == tree.nodes.size());
        for(int i = 0; i < n_nodes; i++){
            Tree::TreeNode node = tree.nodes.host_data()[i];
            children_left[i] = node.lch_index;
            children_right[i] = node.rch_index;
            if(node.default_right)
                children_default[i] = node.rch_index;
            else
                children_default[i] = node.lch_index;
            if(node.is_leaf){
                children_left[i] = -1;
                children_right[i] = -1;
                children_default[i] = -1;
                values[i] = node.base_weight;
            }
            else{
                values[i] = 0;
            }
            features[i] = node.split_feature_id;
            thresholds[i] = node.split_value;
            node_sample_weights[i] = node.sum_gh_pair.h;
        }
//        for(int i = 0; i < n_nodes; i++){
//            Tree::TreeNode node = tree.nodes.host_data()[i];
//            if (node.is_valid && !node.is_pruned){
//                int node_id = node.final_id;
//                children_left[node_id] = tree.nodes.host_data()[node.lch_index].final_id;
//                children_right[node_id] = tree.nodes.host_data()[node.rch_index].final_id;
//                if(node.default_right)
//                    children_default[node_id] = tree.nodes.host_data()[node.rch_index].final_id;
//                else
//                    children_default[node_id] = tree.nodes.host_data()[node.lch_index].final_id;
//                if(node.is_leaf){
//                    children_left[node_id] = -1;
//                    children_right[node_id] = -1;
//                    children_default[node_id] = -1;
//                    values[node_id] = node.base_weight;
//                } else{
//                    values[node_id] = 0;
//                }
//                features[node_id] = node.split_feature_id;
//                thresholds[node_id] = node.split_value;
//                node_sample_weights[node_id] = node.sum_gh_pair.h;
//            }
//        }

    }

    void model_free(Tree* &model){
        if(model){
            delete []model;
        }
    }
}
