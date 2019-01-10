//
// Created by zeyi on 1/9/19.
//

#include "thundergbm/thundergbm.h"
#include "thundergbm/param.h"
#include <thundergbm/trainer.h>
#include "thundergbm/param_parser.h"

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
    TreeTrainer trainer;
    bool exact_sp_producer = true;
    if(exact_sp_producer)
        rmse = trainer.train_exact(model_param);
    else
        rmse = trainer.train_hist(model_param);
    LOG(INFO) << "RMSE is " << rmse;
}