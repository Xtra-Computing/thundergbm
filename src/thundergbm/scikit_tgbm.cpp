//
// Created by zeyi on 1/12/19.
//

#include <thundergbm/trainer.h>
#include "thundergbm/parser.h"
#include <thundergbm/dataset.h>
#include "thundergbm/predictor.h"

extern "C" {
    void sparse_train_scikit(int row_size, float *val, int *row_ptr, int *col_ptr, float *label,
                             int depth, int n_trees, int n_device, float min_child_weight, float lambda,
                             float gamma, int max_num_bin, int verbose, float column_sampling_rate,
                             int bagging, int n_parallel_trees, float learning_rate, char *obj_type,
                             int num_class, char *out_model_name, char *in_model_name,
                             char *tree_method) {
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
        model_param.num_class = num_class;
//        model_param.path = path;
        model_param.out_model_name = out_model_name;
        model_param.in_model_name = in_model_name;
        model_param.tree_method = tree_method;

        model_param.rt_eps = 1e-6;

        if (!verbose) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        }
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        DataSet train_dataset;
        train_dataset.load_from_sparse(row_size, val, row_ptr, col_ptr, label, model_param);
        TreeTrainer trainer;
        trainer.train(model_param, train_dataset);
    }//end sparse_model_scikit

    void sparse_predict_scikit(int row_size, float *val, int *row_ptr, int *col_ptr, float *label,
                               char *in_model_name, float *y_pred){
        //load model
        vector<vector<Tree>> boosted_model;
        GBMParam model_param;
        model_param.in_model_name = in_model_name;
        model_param.path = "../dataset/test_dataset.txt";
        Parser parser;
        parser.load_model(model_param, boosted_model);

        DataSet dataSet;
        dataSet.load_from_sparse(row_size, val, row_ptr, col_ptr, label, model_param);


        //predict
        Predictor pred;
        vector<float_type > y_pred_vec = pred.predict(model_param, boosted_model, dataSet);
        for(int i = 0; i < y_pred_vec.size(); i++)
            y_pred[i] = y_pred_vec[i];
    }
}