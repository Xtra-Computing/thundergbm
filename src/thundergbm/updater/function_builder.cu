//
// Created by ss on 19-1-19.
//

#include <thundergbm/updater/function_builder.h>
#include "thundergbm/updater/exact_tree_builder.h"
#include "thundergbm/updater/hist_tree_builder.h"

FunctionBuilder *FunctionBuilder::create(std::string name) {
    if (name == "exact") return new ExactTreeBuilder;
    if (name == "hist") return new HistTreeBuilder;
    LOG(FATAL) << "unknown builder " << name;
    return nullptr;
}

