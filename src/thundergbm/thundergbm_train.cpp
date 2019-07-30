//
// Created by zeyi on 1/9/19.
//

#include <thundergbm/trainer.h>
#include "thundergbm/parser.h"
#ifdef _WIN32
    INITIALIZE_EASYLOGGINGPP
#endif
int main(int argc, char **argv) {
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);

    GBMParam model_param;
    Parser parser;
    parser.parse_param(model_param, argc, argv);
    if (!model_param.verbose) {
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
    }
    if (!model_param.profiling) {
        el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
    }

    DataSet dataset;
    vector<vector<Tree>> boosted_model;

//    dataset.load_csc_from_file(model_param.path, model_param);
    dataset.load_from_file(model_param.path, model_param);
    TreeTrainer trainer;
    boosted_model = trainer.train(model_param, dataset);
    parser.save_model("tgbm.model", model_param, boosted_model, dataset);
}