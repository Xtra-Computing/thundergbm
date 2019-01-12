//
// Created by jiashuai on 17-9-15.
//
#include <thundergbm/parser.h>
#include "thundergbm/thundergbm.h"
#include "gtest/gtest.h"
#include "thundergbm/param.h"

GBMParam global_test_param;

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
    Parser parser;
    parser.parse_param(global_test_param, argc, argv);
    return RUN_ALL_TESTS();
}
