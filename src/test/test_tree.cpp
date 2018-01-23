//
// Created by jiashuai on 18-1-18.
//

#include <thundergbm/syncarray.h>
#include <thundergbm/tree.h>
#include "gtest/gtest.h"

TEST(TreeTest, test_log) {
    Tree tree(2);
    LOG(DEBUG) << tree.nodes;
}

