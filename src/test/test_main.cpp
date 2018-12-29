//
// Created by jiashuai on 17-9-15.
//
#include "thundergbm/thundergbm.h"
#include "gtest/gtest.h"
#include "thundergbm/param.h"

GBMParam global_test_param;

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    el::Loggers::reconfigureAllLoggers(el::ConfigurationType::Format, "%datetime %level %fbase:%line : %msg");
    el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
    el::Loggers::addFlag(el::LoggingFlag::FixedTimeFormat);
    global_test_param.depth = 6;
    global_test_param.n_trees = 40;
    global_test_param.n_device = 1;
    global_test_param.min_child_weight = 1;
    global_test_param.lambda = 1;
    global_test_param.gamma = 1;
    global_test_param.rt_eps = 1e-6;
    global_test_param.max_num_bin = 255;
    global_test_param.verbose = false;
    global_test_param.column_sampling_rate = 1;
    global_test_param.bagging = false;
    for (int i = 0; i < argc; ++i) {
        if (string(argv[i]) == "-d") global_test_param.depth = atoi(argv[++i]);
        if (string(argv[i]) == "-n") global_test_param.n_trees = atoi(argv[++i]);
        if (string(argv[i]) == "-n_gpu") global_test_param.n_device = atoi(argv[++i]);
        if (string(argv[i]) == "-v") global_test_param.verbose = atoi(argv[++i]);
        if (string(argv[i]) == "-dataset") global_test_param.path = argv[++i];
        if (string(argv[i]) == "-bins") global_test_param.max_num_bin = atoi(argv[++i]);
        if (string(argv[i]) == "-cs") global_test_param.column_sampling_rate = atof(argv[++i]);
        if (string(argv[i]) == "-bagging") global_test_param.bagging= atoi(argv[++i]);
    }
    return RUN_ALL_TESTS();
}
