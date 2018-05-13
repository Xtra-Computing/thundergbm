//
// Created by qinbin on 2018/5/9.
//

#include "thundergbm/hist_cut.h"
#include "thundergbm/quantile_sketch.h"
#include "thundergbm/syncarray.h"
#include <sstream>

void HistCut::get_split_points(SparseColumns& columns, InsStat& stats, int max_num_bins, int n_instances, int n_features){
    //SyncArray<quanSketch> quanSketchs(n_features);
    //quanSketch* sketchs = quanSketchs.host_data();
	split_points.resize(n_features);
    quanSketch sketchs[n_features];
    //vector<float_type> split_points;
    //safe factor for better accuracy
    const int kFactor = 8;
    for(int i = 0; i < n_features; i++){
        sketchs[i].Init(n_instances, 1.0 / (max_num_bins * kFactor));
    }
    float_type* val_ptr = columns.csc_val.host_data();
    int* row_ptr = columns.csc_row_ind.host_data();
    int* col_ptr = columns.csc_col_ptr.host_data();
//    std::cout<<"before add"<<std::endl;
#pragma omp parallel for
//    std::cout<<"col:"<<columns.csc_col_ptr.size()<<std::endl;
	for(int i = 0; i < columns.csc_col_ptr.size() - 1; i++){
        for(int j = col_ptr[i + 1] - 1; j >= col_ptr[i]; j--){
            float_type val = val_ptr[j];
            float_type weight = stats.gh_pair.host_data()[row_ptr[j]].h;
		//	std::cout<<"val:"<<val<<" weight:"<<weight<<std::endl;
            sketchs[i].Add(val, weight);
        }
    }
//	std::cout<<"number of level:"<<sketchs[0].numOfLevel<<std::endl;
//    std::cout<<"after add"<<std::endl;
//	std::cout<<"0 summarySize:"<<sketchs[0].summarySize<<std::endl;
//    std::cout<<"0 summaries size:"<<sketchs[0].summaries.size()<<std::endl;
	//SyncArray<summary> summaries(n_features);
    //summary* n_summary = summaries.host_data();
    summary n_summary[n_features];
#pragma omp parallel for
    for(int i = 0; i < n_features; i++){
        summary ts;
        sketchs[i].GetSummary(ts);
//        if(i == 0) std::cout<<"ts entry size:"<<n_summary[i].entry_size<<std::endl;
		n_summary[i].Reserve(max_num_bins * kFactor);
        n_summary[i].Prune(ts, max_num_bins * kFactor);
    }
//	std::cout<<"entry size:"<<n_summary[0].entry_size<<std::endl;
    for(int i = 0; i < n_features; i++){
        summary ts;
        ts.Reserve(max_num_bins);
        ts.Prune(n_summary[i], max_num_bins);
        if(ts.entry_size == 0) break;
        if(ts.entry_size > 1 && ts.entry_size <= 16){
			//std::cout<<"before push"<<std::endl;
            split_points[i].push_back((ts.entries[0].val + ts.entries[1].val) / 2);
            for(int j = 2; j < ts.entry_size; j++){
                float_type mid = (ts.entries[j - 1].val + ts.entries[j].val) / 2;
                if(mid > split_points[i].back()){
                    split_points[i].push_back(mid);
                }
            }
        }
        else{
			
            split_points[i].push_back(ts.entries[1].val);
            for(int j = 2; j < ts.entry_size; j++){
                float_type val = ts.entries[j].val;
                if(val > split_points[i].back())
                    split_points[i].push_back(val);
            }
        }
        float_type max_val = ts.entries[ts.entry_size - 1].val;
        if(max_val > 0)
            split_points[i].push_back(max_val*2 + 1e-5);
        else
            split_points[i].push_back(1e-5);

    }

}
