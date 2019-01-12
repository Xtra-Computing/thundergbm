//
// Created by zeyi on 1/10/19.
//

#include "thundergbm/thundergbm.h"
#include "thundergbm/param.h"
#include "thundergbm/parser.h"
#include <thundergbm/dataset.h>
#include "thundergbm/predictor.h"

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
    //load data set
    DataSet dataSet;
    dataSet.load_from_file(model_param.path);
    //predict
    Predictor pred;
    pred.predict(trees, dataSet);
}
