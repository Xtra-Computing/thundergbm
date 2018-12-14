//
// Created by qinbin on 2018/5/9.
//

#include "thundergbm/hist_cut.h"
#include "thundergbm/quantile_sketch.h"
#include "thundergbm/syncarray.h"
#include <sstream>
#include <omp.h>
#include <thundergbm/hist_cut.h>

#include "thundergbm/util/device_lambda.cuh"

void HistCut::get_cut_points(SparseColumns &columns, InsStat &stats, int max_num_bins, int n_instances) {
    LOG(TRACE) << "get cut points";
    LOG(DEBUG)<<"val = " << columns.csc_val;
    LOG(DEBUG)<<"idx = " << columns.csc_row_idx;
    LOG(DEBUG)<<"ptr = " << columns.csc_col_ptr;
    int n_features = columns.n_column;
//    std::cout<<"n_featrues:"<<n_features<<std::endl;
    vector<quanSketch> sketchs(n_features);
    const int kFactor = 8;
    for (int i = 0; i < n_features; i++) {
        sketchs[i].Init(n_instances, 1.0 / (max_num_bins * kFactor));
    }
    float_type *val_ptr = columns.csc_val.host_data();
    int *row_ptr = columns.csc_row_idx.host_data();
    int *col_ptr = columns.csc_col_ptr.host_data();
    auto stat_gh_ptr = stats.gh_pair.host_data();
//	std::cout<<"before add"<<std::endl;
#pragma omp parallel for
    for (int i = 0; i < columns.csc_col_ptr.size() - 1; i++) {
        for (int j = col_ptr[i + 1] - 1; j >= col_ptr[i]; j--) {
            float_type val = val_ptr[j];
            float_type weight = stat_gh_ptr[row_ptr[j]].h;
            sketchs[i].Add(val, weight);
        }
    }
//    std::cout<<"after add"<<std::endl;
    vector<summary> n_summary(n_features);
//    summary n_summary[n_features];
//	std::cout<<"before prune"<<std::endl;
#pragma omp parallel for
    for (int i = 0; i < n_features; i++) {
        summary ts;
        sketchs[i].GetSummary(ts);
        n_summary[i].Reserve(max_num_bins * kFactor);
        n_summary[i].Prune(ts, max_num_bins * kFactor);
    }
    int nthread = omp_get_max_threads();
//    LOG(DEBUG)<<"nthread = " << nthread;
    vector<vector<float_type>> cut_points_local;
    cut_points_local.resize(n_features);
    vector<int> cut_points_size(n_features);
    for (int i = 0; i < n_features; i++)
        cut_points_local[i].resize(max_num_bins);
#pragma omp parallel num_threads(nthread)
    {
        int tid = omp_get_thread_num();
//        LOG(DEBUG)<<"tid = "<< tid;
        int nstep = (n_features + nthread - 1) / nthread;
        int sbegin = std::min(tid * nstep, n_features);
        int send = std::min((tid + 1) * nstep, n_features);
        for (int i = sbegin; i < send; i++) {
            int k = 0;
            summary ts;
            ts.Reserve(max_num_bins);
            ts.Prune(n_summary[i], max_num_bins);
            if (ts.entry_size == 0) {
                cut_points_size[i] = 0;
                continue;
            }
            float_type min_val = ts.entries[0].val;

            cut_points_local[i][k++] = min_val - (fabsf(min_val) + 1e-5);

            if (ts.entry_size > 1 && ts.entry_size <= 16) {
                cut_points_local[i][k++] = (ts.entries[0].val + ts.entries[1].val) / 2;
                for (int j = 2; j < ts.entry_size; j++) {
                    float_type mid = (ts.entries[j - 1].val + ts.entries[j].val) / 2;
                    if (mid > cut_points_local[i][k - 1]) {
                        cut_points_local[i][k++] = mid;
                    }
                }
            } else {
                if (ts.entry_size > 1)
                    cut_points_local[i][k++] = ts.entries[1].val;
                for (int j = 2; j < ts.entry_size; j++) {
                    float_type val = ts.entries[j].val;
                    if (val > cut_points_local[i][k - 1]) {
                        cut_points_local[i][k++] = val;
                    }
                }
            }

            /*
            float_type max_val = ts.entries[ts.entry_size - 1].val;
            if(max_val > 0){
                cut_points_local[i][k++] = max_val*2 + 1e-5;
            }
            else{
                cut_points_local[i][k++] = 1e-5;
            }
            */
            cut_points_size[i] = k;
        }
    }
    for (int i = 0; i < n_features; i++) {
        if (cut_points_size[i] != 0)
            this->cut_points.insert(cut_points.end(), cut_points_local[i].begin(),
                                    cut_points_local[i].begin() + cut_points_size[i]);
    }
    this->row_ptr.push_back(0);
    for (int i = 0; i < n_features; i++) {
        this->row_ptr.push_back(cut_points_size[i] + this->row_ptr.back());
    }
    cut_row_ptr.resize(this->row_ptr.size());
    cut_row_ptr.copy_from(this->row_ptr.data(), this->row_ptr.size());
    cut_points_val.resize(this->cut_points.size());
    auto cut_points_val_ptr = cut_points_val.host_data();
    auto cut_row_ptr_data = cut_row_ptr.host_data();
    for (int i = 0; i < cut_row_ptr.size(); i++) {
        int sum = cut_row_ptr_data[i] + cut_row_ptr_data[i + 1] - 1;
        for (int j = cut_row_ptr_data[i + 1] - 1; j >= cut_row_ptr_data[i]; j--)
            cut_points_val_ptr[j] = this->cut_points[sum - j];
    }
    LOG(DEBUG) << cut_row_ptr;
    LOG(DEBUG) << cut_fid.size();
    cut_fid.resize(cut_points.size());
    auto cut_fid_data = cut_fid.device_data();
    device_loop_2d(n_features, cut_row_ptr.device_data(), [=] __device__(int fid, int i) {
        cut_fid_data[i] = fid;
    });
}

void HistCut::get_cut_points2(SparseColumns &columns, InsStat &stats, int max_num_bins, int n_instances) {
    int n_column = columns.n_column;
    auto csc_val_data = columns.csc_val.host_data();
    auto csc_col_ptr_data = columns.csc_col_ptr.host_data();
    cut_points.clear();
    row_ptr.clear();
    row_ptr.resize(1, 0);

    for (int fid = 0; fid < n_column; ++fid) {
        int col_start = csc_col_ptr_data[fid];
        int col_len = csc_col_ptr_data[fid + 1] - col_start;
        auto val_data = csc_val_data + col_start;
        if (col_len <= max_num_bins){
            row_ptr.push_back(col_len + row_ptr.back());
            for (int i = 0; i < col_len; ++i) {
                cut_points.push_back(val_data[i]);
            }
        } else {
            row_ptr.push_back(max_num_bins + row_ptr.back());
            for (int i = 0; i < max_num_bins; ++i) {
                cut_points.push_back(val_data[col_len / 256 * i]);
            }
        }
    }

    cut_points_val.resize(cut_points.size());
    cut_points_val.copy_from(cut_points.data(), cut_points.size());
    cut_row_ptr.resize(row_ptr.size());
    cut_row_ptr.copy_from(row_ptr.data(), row_ptr.size());
    cut_fid.resize(cut_points.size());
    auto cut_fid_data = cut_fid.device_data();
    device_loop_2d(n_column, cut_row_ptr.device_data(), [=] __device__(int fid, int i) {
        cut_fid_data[i] = fid;
    });
}
