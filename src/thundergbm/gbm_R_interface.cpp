// Created by Qinbin on 3/15/2020

#include <thundergbm/trainer.h>
#include "thundergbm/parser.h"
#include "thundergbm/predictor.h"
using std::fstream;
using std::stringstream;

extern "C" {
    void train_R(int* depth, int* n_trees, int* n_gpus, int* verbose,
            int* profiling, char** data, int* max_num_bin, double* column_sampling_rate,
            int* bagging, int* n_parallel_trees, double* learning_rate, char** objective,
            int* num_class, int* min_child_weight, double* lambda_tgbm, double* gamma,
            char** tree_method, char** model_out) {

        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
        el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
        el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

        if (*verbose == 0) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
        } else if (*verbose == 1) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        }

        if (!(*profiling)) {
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        }

        GBMParam model_param;
        model_param.depth = *depth;
        model_param.n_trees = *n_trees;
        model_param.n_device = *n_gpus;
        model_param.verbose = *verbose;
        model_param.profiling = *profiling;
        model_param.path = data[0];
        model_param.max_num_bin = *max_num_bin;
        model_param.column_sampling_rate = (float_type) *column_sampling_rate;
        model_param.bagging = *bagging;
        model_param.n_parallel_trees = *n_parallel_trees;
        model_param.learning_rate = (float_type) * learning_rate;
        model_param.objective = objective[0];
        model_param.num_class = *num_class;
        model_param.min_child_weight = *min_child_weight;
        model_param.lambda = (float_type) *lambda_tgbm;
        model_param.gamma = (float_type) *gamma;
        model_param.tree_method = tree_method[0];
        model_param.rt_eps = 1e-6;
        model_param.tree_per_rounds = 1;

        DataSet dataset;
        Parser parser;
        vector<vector<Tree>> boosted_model;
        dataset.load_from_file(model_param.path, model_param);
        TreeTrainer trainer;
        boosted_model = trainer.train(model_param, dataset);
        parser.save_model(model_out[0], model_param, boosted_model, dataset);
    }

    void predict_R(char** data, char** model_in, int* verbose){
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
        el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
        el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

        if (*verbose == 0) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "false");
        } else if (*verbose == 1) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
        }

        GBMParam model_param;
        Parser parser;
        DataSet dataSet;
        vector<vector<Tree>> boosted_model;
        parser.load_model(model_in[0], model_param, boosted_model, dataSet);
        dataSet.load_from_file(data[0], model_param);
        //predict
        Predictor pred;
        vector<float_type> y_pred_vec = pred.predict(model_param, boosted_model, dataSet);
    }
}
