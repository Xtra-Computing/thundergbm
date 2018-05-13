//
// Created by qinbin on 2018/5/9.
//
#include "gtest/gtest.h"
#include "thundergbm/dataset.h"
#include "thundergbm/hist_cut.h"
#include <thundergbm/tree.h>
#include <thrust/reduce.h>
#include "thrust/adjacent_difference.h"
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <iostream>
//#include "thundergbm/util/device_lambda.cuh"

TEST(TestHistCut, split_points) {
    DataSet dataSet;
    dataSet.load_from_file(DATASET_DIR "abalone");
    //    dataSet.load_from_file(DATASET_DIR "mnist.scale");
    SparseColumns sparseColumns(dataSet);
    //std::cout<<"after columns"<<std::endl;
    InsStat stats;
    int n_instances = dataSet.n_instances();
    stats.init(n_instances);
    //std::cout<<"after stats init"<<std::endl;
    SyncArray<float_type> y(dataSet.y().size());
    y.copy_from(dataSet.y().data(), y.size());
    float_type *y_data = y.host_data();
    GHPair *gh_pair_data = stats.gh_pair.host_data();
    int *nid_data = stats.nid.host_data();
    float_type *stats_y_data = stats.y.host_data();
    float_type *stats_yp_data = stats.y_predict.host_data();
    //LOG(DEBUG) << stats.y_predict;
#pragma omp parallel for
    for(int i = 0; i < n_instances; i++) {
        nid_data[i] = 0;
        stats_y_data[i] = y_data[i];
        gh_pair_data[i].g = stats_yp_data[i] - y_data[i];
        gh_pair_data[i].h = 1;
    };
    //std::cout<<"after stats update"<<std::endl;
    HistCut cut;
    int max_num_bin = 256;
    //std::cout<<"before split"<<std::endl;
    cut.get_split_points(sparseColumns, stats, max_num_bin, dataSet.n_instances(), dataSet.n_features());
    //std::cout<<"points size:"<<cut.split_points.size()<<std::endl;
	//std::cout<<"points:"<<std::endl;
    //for(int i = 0; i < cut.split_points.size(); i++){
    //	std::cout<<"feature:"<<i<<std::endl;
	//	for(int j = 0; j < cut.split_points[i].size();j++){
	//    	std::cout<<cut.split_points[i][j]<<" ";
	//	}
	//	std::cout<<std::endl;
    //}
    CHECK(cut.split_points[0][0] == 1.5) << cut.split_points[0][0];
	CHECK(cut.split_points[0][1] == 2.5) << cut.split_points[0][1];
	CHECK(fabs(cut.split_points[0][2] -  6.00001) < 1e-5 ) << cut.split_points[0][2]; 
    //LOG(DEBUG)<<cut.split_points;
    //    LOG(DEBUG)<<sparseColumns.csc_val;
}
