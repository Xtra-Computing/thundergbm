//
// Created by jiashuai on 17-9-15.
//
#include <thundergbm/parser.h>
#include "gtest/gtest.h"
#ifdef _WIN32
    INITIALIZE_EASYLOGGINGPP
#endif
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
    return RUN_ALL_TESTS();
}
