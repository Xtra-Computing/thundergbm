//
// Created by jiashuai on 18-1-18.
//
#include "gtest/gtest.h"
#include "thundergbm/dataset.h"

TEST(TestSparseColums, dataset_loading) {
    DataSet dataSet;
    dataSet.load_from_file(DATASET_DIR "test_dataset.txt");
//    dataSet.load_from_file(DATASET_DIR "mnist.scale");
    SparseColumns sparseColumns(dataSet);
}
