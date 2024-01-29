//
// Created by ss on 19-1-20.
//
#include "thundergbm/builder/hist_tree_builder_single.h"

#include "thundergbm/util/cub_wrapper.h"
#include "thundergbm/util/device_lambda.cuh"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/discard_iterator.h"
#include "thrust/sequence.h"
#include "thrust/binary_search.h"
#include "thundergbm/util/multi_device.h"

#include "omp.h"
#include "cusparse.h"
#include <chrono>
#include<iostream>
typedef std::chrono::high_resolution_clock Clock;
#define TDEF(x_) std::chrono::high_resolution_clock::time_point x_##_t0, x_##_t1;
#define TSTART(x_) x_##_t0 = Clock::now();
#define TEND(x_) x_##_t1 = Clock::now();
#define TPRINT(x_, str) printf("%-20s \t%.6f\t sec\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()/1e6);
#define TINT(x_) std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()

long long total_hist_time;
long long total_evaluate_time;


void check_hist_res(GHPair* hist, GHPair* hist_test, int n_bins){

    //check result
    float avg_diff_g = 0;
    float total_diff_g= 0;
    
    for(int i = 0;i<n_bins;++i){
        
        total_diff_g += abs(hist_test[i].g-hist[i].g);
    
    }
    avg_diff_g = total_diff_g/n_bins;
    
    LOG(INFO)<<"total diff g is "<<total_diff_g<<" avg diff g is "<<avg_diff_g;

}



void HistTreeBuilder_single::get_bin_ids() {
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        SparseColumns &columns = shards[device_id].columns;
        HistCut &cut = this->cut[device_id];
        //auto &dense_bin_id = this->dense_bin_id[device_id];
        using namespace thrust;
        int n_column = columns.n_column;
        size_t nnz = columns.nnz;
        auto cut_row_ptr = cut.cut_row_ptr.device_data();
        auto cut_points_ptr = cut.cut_points_val.device_data();
        
        int n_block = fminf((nnz / n_column - 1) / 256 + 1, 6 * 84);
        //original order csc
        //auto csc_val_origin_data = columns.csc_val_origin.device_data();
        
        //auto &bin_id_origin = this->bin_id_origin[device_id];
        //bin_id_origin.resize(columns.nnz);
        //auto bin_id_origin_data = bin_id_origin.device_data();
        
        auto &csr_row_ptr = this->csr_row_ptr[device_id];
        auto &csr_col_idx = this->csr_col_idx[device_id];
        auto &csr_bin_id  = this->csr_bin_id[device_id];

        //set poniter
        csr_row_ptr.resize(n_instances+1);
        csr_col_idx.resize(nnz);
        csr_bin_id.resize(nnz);
        
        csr_row_ptr.set_device_data(columns.csr_row_ptr.device_data());
        csr_col_idx.set_device_data(columns.csr_col_idx.device_data());

        {   //in thrust lowerbound <= first element, upperbound is < first element 
            auto upperBound = [=]__device__(const float_type *search_begin, const float_type *search_end, float_type val) {
                const float_type *left = search_begin;
                const float_type *right = search_end - 1;

                while (left < right) {
                    const float_type *mid = left + (right - left) / 2;
                    if ((float)*mid <= (float)val)
                        left = mid+1;
                    else right = mid;
                }
                return left;
            };
            TIMED_SCOPE(timerObj, "binning");
            //for original order csc
            //device_loop_2d(n_column, columns.csc_col_ptr_origin.device_data(), [=]__device__(int cid, int i) {
            //    auto search_begin = cut_points_ptr + cut_row_ptr[cid];
            //    auto search_end = cut_points_ptr + cut_row_ptr[cid + 1];
            //    auto val = csc_val_origin_data[i];
            //    bin_id_origin_data[i] = lowerBound(search_begin, search_end, val) - search_begin;
            //}, n_block);

            //get csr bin id
            auto csr_col_idx_data = csr_col_idx.device_data();
            auto csr_val_data = columns.csr_val.device_data();
            auto csr_bin_id_data = csr_bin_id.device_data();
            //device_loop_2d(n_instances, csr_row_ptr.device_data(), [=]__device__(int instance_id, int i) {
            //    auto cid = csr_col_idx_data[i];
            //    auto search_begin = cut_points_ptr + cut_row_ptr[cid];
            //    auto search_end = cut_points_ptr + cut_row_ptr[cid + 1];
            //    auto val = csr_val_data[i];
            //    csr_bin_id_data[i] = upperBound(search_begin, search_end, val) - search_begin + cut_row_ptr[cid];
            //}, n_block);
            device_loop(nnz, [=]__device__( int i) {
                auto cid = csr_col_idx_data[i];
                auto search_begin = cut_points_ptr + cut_row_ptr[cid];
                auto search_end = cut_points_ptr + cut_row_ptr[cid + 1];
                auto val = csr_val_data[i];
                csr_bin_id_data[i] = upperBound(search_begin, search_end, val) - search_begin + cut_row_ptr[cid];
            });

        }
        //columns.csc_val_origin.clear_device();
        //columns.csc_val_origin.resize(0);
        columns.csr_val.resize(0);
        //csr col idx do not need
        columns.csr_col_idx.resize(0);
        SyncMem::clear_cache();

    });
}

void HistTreeBuilder_single::find_split(int level, int device_id) {
    std::chrono::high_resolution_clock timer;

    const SparseColumns &columns = shards[device_id].columns;
    SyncArray<int> &nid = ins2node_id[device_id];
    SyncArray<GHPair> &gh_pair = gradients[device_id];
    Tree &tree = trees[device_id];
    SyncArray<SplitPoint> &sp = this->sp[device_id];
    SyncArray<bool> &ignored_set = shards[device_id].ignored_set;
    HistCut &cut = this->cut[device_id];
    //auto &dense_bin_id = this->dense_bin_id[device_id];
    auto &last_hist = this->last_hist[device_id];
    int max_trick_depth = columns.max_trick_depth;
    int max_trick_nodes = columns.max_trick_nodes;
    
    TIMED_FUNC(timerObj);
    int n_nodes_in_level = static_cast<int>(pow(2, level));
    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = columns.n_column;
    int n_bins = cut.cut_points_val.size();
    //max nodes needs to build histograms
    int n_max_nodes = 1 << (param.depth-1);
    size_t n_split = n_nodes_in_level * (long long)n_bins;

    int n_block = fminf((columns.nnz / this->n_instances - 1) / 256 + 1, 4 * 84);
    //int avg_f = (columns.nnz / this->n_instances );
    //csr bin id
    auto &csr_row_ptr = this->csr_row_ptr[device_id];
    auto &csr_col_idx = this->csr_col_idx[device_id];
    auto &csr_bin_id  = this->csr_bin_id[device_id];
    
    auto csr_row_ptr_data = csr_row_ptr.device_data();
    auto csr_col_idx_data = csr_col_idx.device_data();
    auto csr_bin_id_data = csr_bin_id.device_data();
    LOG(TRACE) << "start finding split";
    TDEF(hist)
    TDEF(evaluate)

    //new variables
    size_t len_hist = 2*n_bins;
    size_t len_missing = 2*n_column;

    //remember resize variable to clear
    //find the best split locally
    {
        using namespace thrust;
        auto t_build_start = timer.now();

        //calculate split information for each split
        SyncArray<GHPair> hist(len_hist);
        SyncArray<GHPair> missing_gh(len_missing);
        auto cut_fid_data = cut.cut_fid.device_data();
        auto i2fid = [=] __device__(int i) { return cut_fid_data[i % n_bins]; };
        auto hist_fid = make_transform_iterator(counting_iterator<int>(0), i2fid);
        {
            {
                TIMED_SCOPE(timerObj, "build hist");
                {
                    size_t
                    smem_size = n_bins * sizeof(GHPair);
                    LOG(DEBUG) << "shared memory size = " << smem_size / 1024.0 << " KB";
                    if (n_nodes_in_level == 1) {
                        //root

                        TSTART(hist)

                        auto hist_data = hist.device_data();
                        auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
                        auto gh_data = gh_pair.device_data();
                        //auto dense_bin_id_data = dense_bin_id.device_data();
                        auto n_instances = this->n_instances;
                        device_loop_hist_csr_root(n_instances,csr_row_ptr_data, [=]__device__(int i,int j){
                        
                            //int fid = csr_col_idx_data[j];
                            int bid = (int)csr_bin_id_data[j];
                            //int feature_offset = cut_row_ptr_data[fid];
                            const GHPair src = gh_data[i];
                            //GHPair &dest = hist_data[feature_offset + bid]; 
                            GHPair &dest = hist_data[bid]; 
                            if(src.h != 0)
                                atomicAdd(&dest.h, src.h);
                            if(src.g != 0)
                                atomicAdd(&dest.g, src.g);                            
                        
                        },n_block);

                        TEND(hist)
                        total_hist_time+=TINT(hist);
                        
                        //LOG(INFO)<<"root hist feature 4447 last bin"<<hist.host_data()[cut.cut_row_ptr.host_data()[4448]-1];
                        //LOG(INFO)<<"root hist feature 4447 last index is "<<cut.cut_row_ptr.host_data()[4448]-1;
                        
                        //device_loop(n_instances,[=]__device__(int i){
                        //    auto st = csr_row_ptr_data[i];
                        //    auto ed = csr_row_ptr_data[i+1];
                        //    const GHPair src = gh_data[i];
                        //    for(int j = st;j<ed;j++){
                        //        int bid = (int)csr_bin_id_data[j];
                        //        GHPair &dest = hist_data[bid];
                        //        if(src.h != 0)
                        //            atomicAdd(&dest.h, src.h);
                        //        if(src.g != 0)
                        //            atomicAdd(&dest.g, src.g);
                        //    }
                        //
                        //});
                        //device_loop(csr_bin_id.size(),[=]__device__(int i){
                        //    int bid = csr_bin_id_data[i];
                        //    int iid = csr_col_idx_data[i];
                        //    const GHPair src = gh_data[iid];
                        //    GHPair &dest = hist_data[bid];
                        //    if(src.h != 0)
                        //        atomicAdd(&dest.h, src.h);
                        //    if(src.g != 0)
                        //        atomicAdd(&dest.g, src.g);
                        //});

                        TSTART(evaluate)

                        //new code 
                        last_hist.copy_from(hist.device_data(),n_bins);
                        cudaDeviceSynchronize();
                        


                        inclusive_scan_by_key(cuda::par, hist_fid, hist_fid + n_bins,
                                      hist.device_data(), hist.device_data());

                        { //missing 
                            auto nodes_data = tree.nodes.device_data();
                            auto missing_gh_data = missing_gh.device_data();
                            auto cut_row_ptr = cut.cut_row_ptr.device_data();
                            auto hist_data = hist.device_data();
                            device_loop(n_column, [=]__device__(int pid) {
                                int nid0 = pid / n_column;
                                int nid = nid0 + nid_offset;
                                if (!nodes_data[nid].splittable()) return;
                                int fid = pid % n_column;
                                if (cut_row_ptr[fid + 1] != cut_row_ptr[fid]) {
                                    GHPair node_gh = hist_data[nid0 * n_bins + cut_row_ptr[fid + 1] - 1];
                                    missing_gh_data[pid] = nodes_data[nid].sum_gh_pair - node_gh;
                                }
                            });

                        }

                        //
                        SyncArray<float_type> gain(n_bins);
                        {
                //            TIMED_SCOPE(timerObj, "calculate gain");
                            auto compute_gain = []__device__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                    float_type lambda) -> float_type {
                                    if (lch.h >= min_child_weight && rch.h >= min_child_weight)
                                    return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda)
                                                -(father.g * father.g) / (father.h + lambda);
                                    else
                                    return 0;
                            };

                            const Tree::TreeNode *nodes_data = tree.nodes.device_data();
                            GHPair *gh_prefix_sum_data = hist.device_data();
                            float_type *gain_data = gain.device_data();
                            const auto missing_gh_data = missing_gh.device_data();
                            auto ignored_set_data = ignored_set.device_data();
                            auto cut_row_ptr = cut.cut_row_ptr.device_data();
                            //for lambda expression
                            float_type mcw = param.min_child_weight;
                            float_type l = param.lambda;
                            device_loop(n_bins, [=]__device__(int i) {
                                int nid0 = i / n_bins;
                                int nid = nid0 + nid_offset;
                                int fid = hist_fid[i % n_bins];
                                if (nodes_data[nid].is_valid && !ignored_set_data[fid]) {
                                    int pid = nid0 * n_column + hist_fid[i];
                                    GHPair father_gh = nodes_data[nid].sum_gh_pair;
                                    GHPair p_missing_gh = missing_gh_data[pid];
                                    GHPair lch_gh = GHPair(0);
                                    if(cut_row_ptr[fid]!=i)
                                        lch_gh = gh_prefix_sum_data[i-1];
                                    float_type default_to_right_gain = max(0.f,
                                                                          compute_gain(father_gh, lch_gh, father_gh - lch_gh, mcw, l));
                                    lch_gh = lch_gh + p_missing_gh;
                                    float_type default_to_left_gain = max(0.f,
                                                                           compute_gain(father_gh, lch_gh, father_gh - lch_gh, mcw, l));
                                    if (default_to_left_gain > default_to_right_gain)
                                        gain_data[i] = default_to_left_gain;
                                    else
                                        gain_data[i] = -default_to_right_gain;//negative means default split to right


                                } else gain_data[i] = 0;
                            });
                            LOG(DEBUG) << "gain = " << gain;
                        }

                        SyncArray<int_float> best_idx_gain(n_nodes_in_level);
                        {
                //            TIMED_SCOPE(timerObj, "get best gain");
                            auto arg_abs_max = []__device__(const int_float &a, const int_float &b) {
                                if (fabsf(get<1>(a)) == fabsf(get<1>(b)))
                                    return get<0>(a) < get<0>(b) ? a : b;
                                else
                                    return fabsf(get<1>(a)) > fabsf(get<1>(b)) ? a : b;
                            };

                            auto nid_iterator = make_transform_iterator(counting_iterator<int>(0), placeholders::_1 / n_bins);

                            reduce_by_key(
                                    cuda::par,
                                    nid_iterator, nid_iterator + n_split,
                                    make_zip_iterator(make_tuple(counting_iterator<int>(0), gain.device_data())),
                                    make_discard_iterator(),
                                    best_idx_gain.device_data(),
                                    thrust::equal_to<int>(),
                                    arg_abs_max
                            );
                            LOG(DEBUG) << n_split;
                            LOG(DEBUG) << "best rank & gain = " << best_idx_gain;
                        }
					
                        {
                            const int_float *best_idx_gain_data = best_idx_gain.device_data();
                            auto hist_data = hist.device_data();
                            const auto missing_gh_data = missing_gh.device_data();
                            auto cut_val_data = cut.cut_points_val.device_data();

                            sp.resize(n_nodes_in_level);
                            auto sp_data = sp.device_data();
                            auto nodes_data = tree.nodes.device_data();

                            int column_offset = columns.column_offset;

                            auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
                            device_loop(n_nodes_in_level, [=]__device__(int i) {
                                int_float bst = best_idx_gain_data[i];
                                float_type best_split_gain = get<1>(bst);
                                int split_index = get<0>(bst);
                                if (!nodes_data[i + nid_offset].is_valid) {
                                    sp_data[i].split_fea_id = -1;
                                    sp_data[i].nid = -1;
                                    return;
                                }
                                int fid = hist_fid[split_index];
                                sp_data[i].split_fea_id = fid + column_offset;
                                sp_data[i].nid = i + nid_offset;
                                sp_data[i].gain = fabsf(best_split_gain);
                                //sp_data[i].fval = cut_val_data[split_index % n_bins];
                                sp_data[i].split_bid = (unsigned char) (split_index % n_bins - cut_row_ptr_data[fid]);
                                sp_data[i].fea_missing_gh = missing_gh_data[i * n_column + hist_fid[split_index]];
                                sp_data[i].default_right = best_split_gain < 0;
                                sp_data[i].lch_sum_gh = GHPair(0);
                                sp_data[i].fval = cut_val_data[cut_row_ptr_data[fid]] - fabsf(cut_val_data[cut_row_ptr_data[fid]])- 1e-5;
                                if(split_index!=cut_row_ptr_data[fid]){
                                    sp_data[i].lch_sum_gh = hist_data[split_index-1];
                                    sp_data[i].fval = cut_val_data[split_index % n_bins-1];
                                    //sp_data[i].split_bid -=1;
                                }
                            });
                        }
                        TEND(evaluate)
                        total_evaluate_time+=TINT(evaluate);
                        
                    } else {
                        //otherwise
                        auto t_dp_begin = timer.now();
                        SyncArray<int> node_idx(n_instances);
                        SyncArray<int> node_ptr(n_nodes_in_level + 1);
                        {
                            TIMED_SCOPE(timerObj, "data partitioning");
                            SyncArray<int> nid4sort(n_instances);
                            nid4sort.copy_from(ins2node_id[device_id]);
                            sequence(cuda::par, node_idx.device_data(), node_idx.device_end(), 0);

                            cub_sort_by_key(nid4sort, node_idx);
                            auto counting_iter = make_counting_iterator < int > (nid_offset);
                            node_ptr.host_data()[0] =
                                    lower_bound(cuda::par, nid4sort.device_data(), nid4sort.device_end(), nid_offset) -
                                    nid4sort.device_data();

                            upper_bound(cuda::par, nid4sort.device_data(), nid4sort.device_end(), counting_iter,
                                        counting_iter + n_nodes_in_level, node_ptr.device_data() + 1);
                            LOG(DEBUG) << "node ptr = " << node_ptr;
                            cudaDeviceSynchronize();
                        }
                        auto t_dp_end = timer.now();
                        std::chrono::duration<double> dp_used_time = t_dp_end - t_dp_begin;
                        this->total_dp_time += dp_used_time.count();


                        auto node_ptr_data = node_ptr.host_data();
                        auto node_idx_data = node_idx.device_data();
                        auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
                        auto gh_data = gh_pair.device_data();
                        //auto dense_bin_id_data = dense_bin_id.device_data();

                        //new varibales
                        size_t last_hist_len = n_max_nodes/2;
                        size_t half_last_hist_len = n_max_nodes/4;

                        if(max_trick_depth!=-1){
                            last_hist_len = max_trick_nodes*2;
                            half_last_hist_len = max_trick_nodes;

                        }

                        SyncArray<float_type> gain(len_hist);
                        SyncArray<int_float> best_idx_gain(2);
                        sp.resize(n_nodes_in_level);
                        //test
                        // SyncArray<GHPair> test(2*n_bins);
                        // auto test_data = test.device_data();
                        //LOG(INFO)<<"level "<<level<<" nid_offset "<<nid_offset;
                        for (int i = 0; i < n_nodes_in_level / 2; ++i) {
                            size_t tmp_index = i;
                            int nid0_to_compute = i * 2;
                            int nid0_to_substract = i * 2 + 1;
                            int n_ins_left = node_ptr_data[nid0_to_compute + 1] - node_ptr_data[nid0_to_compute];
                            int n_ins_right = node_ptr_data[nid0_to_substract + 1] - node_ptr_data[nid0_to_substract];
                            
                            //LOG(INFO)<<"node index  "<<nid_offset+i*2<<", n_ins_left "<<n_ins_left;
                            //LOG(INFO)<<"node index  "<<nid_offset+i*2+1<<", n_ins_right "<<n_ins_right;
                            
                            if (max(n_ins_left, n_ins_right) == 0) 
                            {   
                                auto nodes_data = tree.nodes.device_data();
                                auto sp_data = sp.device_data();
                                device_loop(2, [=]__device__(int i) {
                                    if (!nodes_data[i + nid_offset+2*tmp_index].is_valid) {
                                        sp_data[i+2*tmp_index].split_fea_id = -1;
                                        sp_data[i+2*tmp_index].nid = -1;
                                    }
                                });
                                continue;
                            }
                            if (n_ins_left > n_ins_right)
                                swap(nid0_to_compute, nid0_to_substract);
                            
                            size_t computed_hist_pos = nid0_to_compute%2;
                            size_t to_compute_hist_pos = 1-computed_hist_pos;

                            TSTART(hist)
                            //compute
                            {
                                int nid0 = nid0_to_compute;
                                auto idx_begin = node_ptr.host_data()[nid0];
                                auto idx_end = node_ptr.host_data()[nid0 + 1];
                                auto hist_data = hist.device_data() + computed_hist_pos*n_bins;
                                this->total_hist_num++;

                                //reset zero
                                cudaMemset(hist_data, 0, n_bins*sizeof(GHPair));

                                //new csr loop
                                device_loop_hist_csr_node((idx_end - idx_begin),csr_row_ptr_data, [=]__device__(int i,int current_pos,int stride){
                                    //iid
                                    int iid = node_idx_data[i+idx_begin];
                                    int begin = csr_row_ptr_data[iid];
                                    int end = csr_row_ptr_data[iid+1];
                                    for(int j = begin+current_pos;j<end;j+=stride){
                                        //int fid = csr_col_idx_data[j];
                                        int bid = (int)csr_bin_id_data[j];

                                        //int feature_offset = cut_row_ptr_data[fid];
                                        const GHPair src = gh_data[iid];
                                        //GHPair &dest = hist_data[feature_offset + bid];
                                        GHPair &dest = hist_data[bid];
                                        if(src.h!= 0){
                                            atomicAdd(&dest.h, src.h);
                                        }
                                        if(src.g!= 0){
                                            atomicAdd(&dest.g, src.g);
                                        }
                                    }
                                },n_block);
                                    //device_loop((idx_end - idx_begin),[=]__device__(int i ){
                                    //    int iid = node_idx_data[i+idx_begin];
                                    //    int begin = csr_row_ptr_data[iid];
                                    //    int end = csr_row_ptr_data[iid+1];
                                    //    const GHPair src = gh_data[iid];
                                    //    for(int j = begin;j<end;j++){
                                    //        int bid = csr_bin_id_data[j];
                                    //        GHPair &dest = hist_data[bid];
                                    //        if(src.h!= 0){
                                    //            atomicAdd(&dest.h, src.h);
                                    //        }
                                    //        if(src.g!= 0){
                                    //            atomicAdd(&dest.g, src.g);
                                    //        }
                                    //    }
                                    //    
                                    //});

                                
                            }
                            if(max_trick_depth<0 || level<=max_trick_depth||(nid0_to_substract / 2)<max_trick_nodes){
                                //subtract
                                {
                                    auto hist_data_computed = hist.device_data() + computed_hist_pos * n_bins;
                                    auto hist_data_to_compute = hist.device_data() + to_compute_hist_pos * n_bins;
                                    auto father_hist_data = last_hist.device_data() + (size_t)(nid0_to_substract / 2) * n_bins;
                                    
                                    if(level%2==0){
                                        size_t st_pos = (((half_last_hist_len+(nid0_to_substract / 2)))%last_hist_len)* n_bins;
                                        father_hist_data = last_hist.device_data() + st_pos ;
                                    }
                                    
                                    device_loop(n_bins, [=]__device__(int i) {
                                        hist_data_to_compute[i] = father_hist_data[i] - hist_data_computed[i];
                                    });
							    	
                                }
                            }
                            else{
                                //compute, if trick is not working
                                int nid0 = nid0_to_substract;
                                auto idx_begin = node_ptr.host_data()[nid0];
                                auto idx_end = node_ptr.host_data()[nid0 + 1];
                                auto hist_data = hist.device_data() + to_compute_hist_pos*n_bins;

                                cudaMemset(hist_data, 0, n_bins*sizeof(GHPair));

                                device_loop_hist_csr_node((idx_end - idx_begin),csr_row_ptr_data, [=]__device__(int i,int current_pos,int stride){
                                    int iid = node_idx_data[i+idx_begin];
                                    int begin = csr_row_ptr_data[iid];
                                    int end = csr_row_ptr_data[iid+1];
                                    for(int j = begin+current_pos;j<end;j+=stride){
                                        int bid = (int)csr_bin_id_data[j];
                                        const GHPair src = gh_data[iid];
                                        GHPair &dest = hist_data[bid];
                                        if(src.h!= 0){
                                            atomicAdd(&dest.h, src.h);
                                        }
                                        if(src.g!= 0){
                                            atomicAdd(&dest.g, src.g);
                                        }
                                    }
                                },n_block);

                            }
                            TEND(hist)
                            total_hist_time+=TINT(hist);
//                            PERFORMANCE_CHECKPOINT(timerObj);

                            //设计last_hist的拷贝策略

                            if((level<(param.depth-1) && max_trick_depth==-1)||(level<max_trick_depth)){
                                if(level%2==0){
                                    //even level
                                    // LOG(INFO)<<"start_pos in even level "<<2*i;
                                    cudaMemcpy(last_hist.device_data()+(size_t)2*i*n_bins, hist.device_data(), 2*n_bins*sizeof(GHPair), cudaMemcpyDefault);
                                }
                                else{
                                    //odd level
                                    //start copy position
                                    size_t start_pos = ((half_last_hist_len+2*i)%last_hist_len)*n_bins;
                                    if(param.depth!=3){
                                        //size_t start_pos = ((half_last_hist_len+2*i)%last_hist_len)*n_bins;
                                        // LOG(INFO)<<"start_pos in odd level "<<((half_last_hist_len+2*i)%last_hist_len);
                                        cudaMemcpy(last_hist.device_data()+start_pos,hist.device_data(), 2*n_bins*sizeof(GHPair), cudaMemcpyDefault);
                                    }else{
                                        cudaMemcpy(last_hist.device_data()+start_pos,hist.device_data(), n_bins*sizeof(GHPair), cudaMemcpyDefault);
                                        
                                        cudaMemcpy(last_hist.device_data(),hist.device_data()+n_bins, n_bins*sizeof(GHPair), cudaMemcpyDefault);
                                    }
                                }
                            }
                            //parent array not enough, part nodes use trick
                            else if(level<(param.depth-1) && level>=max_trick_depth && 2*i<max_trick_nodes){ 
                                if(level%2==0){
                                    cudaMemcpy(last_hist.device_data()+(size_t)2*i*n_bins, hist.device_data(), 2*n_bins*sizeof(GHPair), cudaMemcpyDefault);
                                }
                                else{
                                    size_t start_pos = (half_last_hist_len+2*i)*n_bins;
                                    cudaMemcpy(last_hist.device_data()+start_pos,hist.device_data(), 2*n_bins*sizeof(GHPair), cudaMemcpyDefault);
                                }
                                
                            }


                            cudaDeviceSynchronize();
                            
                            TSTART(evaluate)
                            inclusive_scan_by_key(cuda::par, hist_fid, hist_fid + 2*n_bins,
                                        hist.device_data(), 
                                        hist.device_data());

                        
                            { //missing 
                                auto nodes_data = tree.nodes.device_data();
                                auto missing_gh_data = missing_gh.device_data();
                                auto cut_row_ptr = cut.cut_row_ptr.device_data();
                                auto hist_data = hist.device_data();
                                int loop_len = n_column*2;
                                device_loop(loop_len, [=]__device__(int pid) {
                                    int nid0 = (pid / n_column);
                                    int nid = nid0 + nid_offset+2*tmp_index;
                                    if (!nodes_data[nid].splittable()) return;
                                    int fid = pid % n_column;
                                    if (cut_row_ptr[fid + 1] != cut_row_ptr[fid]) {
                                        GHPair node_gh = hist_data[nid0 * n_bins + cut_row_ptr[fid + 1] - 1];
                                        missing_gh_data[pid] = nodes_data[nid].sum_gh_pair - node_gh;
                                    }
                                });

                            }

                            {
                    //            TIMED_SCOPE(timerObj, "calculate gain");
                                auto compute_gain = []__device__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                        float_type lambda) -> float_type {
                                        if (lch.h >= min_child_weight && rch.h >= min_child_weight)
                                        return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda)
                                                -(father.g * father.g) / (father.h + lambda);
                                        else
                                        return 0;
                                };

                                const Tree::TreeNode *nodes_data = tree.nodes.device_data();
                                GHPair *gh_prefix_sum_data = hist.device_data();
                                float_type *gain_data = gain.device_data();
                                const auto missing_gh_data = missing_gh.device_data();
                                auto ignored_set_data = ignored_set.device_data();
                                auto cut_row_ptr = cut.cut_row_ptr.device_data();
                                
                                //for lambda expression
                                float_type mcw = param.min_child_weight;
                                float_type l = param.lambda;
                                device_loop(2*n_bins, [=]__device__(int i) {
                                    int nid0 = i / n_bins;
                                    int nid = nid0 + nid_offset+2*tmp_index;
                                    int fid = hist_fid[i % n_bins];
                                    if (nodes_data[nid].is_valid && !ignored_set_data[fid]) {
                                        int pid = nid0 * n_column + hist_fid[i];
                                        GHPair father_gh = nodes_data[nid].sum_gh_pair;
                                        GHPair p_missing_gh = missing_gh_data[pid];
                                        GHPair lch_gh = GHPair(0);
                                        if(nid0*n_bins+cut_row_ptr[fid]!=i)
                                            lch_gh = gh_prefix_sum_data[i-1];
                                        float_type default_to_right_gain = max(0.f,
                                                                              compute_gain(father_gh, lch_gh, father_gh - lch_gh, mcw, l));
                                        lch_gh = lch_gh + p_missing_gh;
                                        float_type default_to_left_gain = max(0.f,
                                                                               compute_gain(father_gh, lch_gh, father_gh - lch_gh, mcw, l));
                                        if (default_to_left_gain > default_to_right_gain)
                                            gain_data[i] = default_to_left_gain;
                                        else
                                            gain_data[i] = -default_to_right_gain;//negative means default split to right

                                    } else gain_data[i] = 0;
                                });
                                LOG(DEBUG) << "gain = " << gain;
                            }
                            
                            {
                    //            TIMED_SCOPE(timerObj, "get best gain");
                                auto arg_abs_max = []__device__(const int_float &a, const int_float &b) {
                                    if (fabsf(get<1>(a)) == fabsf(get<1>(b)))
                                        return get<0>(a) < get<0>(b) ? a : b;
                                    else
                                        return fabsf(get<1>(a)) > fabsf(get<1>(b)) ? a : b;
                                };

                                auto nid_iterator = make_transform_iterator(counting_iterator<int>(0), placeholders::_1 / n_bins);

                                reduce_by_key(
                                        cuda::par,
                                        nid_iterator, nid_iterator + 2*n_bins,
                                        make_zip_iterator(make_tuple(counting_iterator<int>(0), gain.device_data())),
                                        make_discard_iterator(),
                                        best_idx_gain.device_data(),
                                        thrust::equal_to<int>(),
                                        arg_abs_max
                                );
                                LOG(DEBUG) << "best rank & gain = " << best_idx_gain;
                            }

                            //get split points
                            {
                                const int_float *best_idx_gain_data = best_idx_gain.device_data();
                                auto hist_data = hist.device_data();
                                const auto missing_gh_data = missing_gh.device_data();
                                auto cut_val_data = cut.cut_points_val.device_data();

                                
                                auto sp_data = sp.device_data();
                                auto nodes_data = tree.nodes.device_data();

                                int column_offset = columns.column_offset;

                                auto cut_row_ptr_data = cut.cut_row_ptr.device_data();
                                
                                device_loop(2, [=]__device__(int i) {
                                    int_float bst = best_idx_gain_data[i];
                                    float_type best_split_gain = get<1>(bst);
                                    size_t split_index = get<0>(bst);
                                    if (!nodes_data[i + nid_offset+2*tmp_index].is_valid) {
                                        sp_data[i+2*tmp_index].split_fea_id = -1;
                                        sp_data[i+2*tmp_index].nid = -1;
                                        return;
                                    }
                                    int fid = hist_fid[split_index];
                                    sp_data[i+2*tmp_index].split_fea_id = fid + column_offset;
                                    sp_data[i+2*tmp_index].nid = i + nid_offset+2*tmp_index;
                                    sp_data[i+2*tmp_index].gain = fabsf(best_split_gain);
                                    //sp_data[i+2*tmp_index].fval = cut_val_data[split_index % n_bins];
                                    sp_data[i+2*tmp_index].split_bid = (unsigned char) (split_index % n_bins - cut_row_ptr_data[fid]);
                                    sp_data[i+2*tmp_index].fea_missing_gh = missing_gh_data[(i) * n_column + hist_fid[split_index]];
                                    sp_data[i+2*tmp_index].default_right = best_split_gain < 0;
                                   
                                    sp_data[i+2*tmp_index].lch_sum_gh = GHPair(0);
                                    sp_data[i+2*tmp_index].fval = cut_val_data[cut_row_ptr_data[fid]] - fabsf(cut_val_data[cut_row_ptr_data[fid]])- 1e-5;
                                    //TODO improve implementation
                                    if(split_index!=i*n_bins+cut_row_ptr_data[fid]){
                                        sp_data[i+2*tmp_index].lch_sum_gh = hist_data[i*n_bins+split_index%n_bins-1];
                                        sp_data[i+2*tmp_index].fval = cut_val_data[split_index % n_bins-1];
                                        //sp_data[i+2*tmp_index].split_bid -=1;
                                    }

                                });
                                
                            }

                           TEND(evaluate)
                           total_evaluate_time+=TINT(evaluate);


                        }  // end for each node

                        //clear array
                        //hist.resize(0);
                        //missing_gh.resize(0);
                        //gain.resize(0);
                        //best_idx_gain.resize(0);
                        
                            
                    }//end # node > 1
                    
                }
                
            }
        }
        //calculate gain of each split

        
        
    }

    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}


void HistTreeBuilder_single::init(const DataSet &dataset, const GBMParam &param) {
    TreeBuilder::init(dataset, param);
    //TODO refactor
    //init shards
    int n_device = param.n_device;
    shards = vector<Shard>(n_device);
    vector<std::unique_ptr<SparseColumns>> v_columns(param.n_device);
    for (int i = 0; i < param.n_device; ++i) {
        v_columns[i].reset(&shards[i].columns);
        shards[i].ignored_set = SyncArray<bool>(dataset.n_features());
    }
    SparseColumns columns;
    
    columns.to_gpu(dataset, v_columns);
    
    cut = vector<HistCut>(param.n_device);
    //dense_bin_id = MSyncArray<unsigned char>(param.n_device);
    last_hist = MSyncArray<GHPair>(param.n_device);

    //csr bin id
    csr_bin_id = MSyncArray<int>(param.n_device);
    csr_row_ptr = MSyncArray<int>(param.n_device);
    csr_col_idx = MSyncArray<int>(param.n_device);

    //bin_id_origin = MSyncArray<unsigned char>(param.n_device);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        cut[device_id].get_cut_points_single(shards[device_id].columns, param.max_num_bin, n_instances);
        
        size_t free_byte,total_byte;
        cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
        LOG(INFO)<<"free memory now is "<<free_byte / (1024.0 * 1024.0*1024) << " GB";
        //keep some extra memory
        free_byte = free_byte - (size_t)3*1024*1024*1024;
        
        size_t need_size = (1 << (param.depth-2)) * cut[device_id].cut_points_val.size();
        LOG(INFO)<<"last hist size is "<<need_size*8/(1024*1024*1024.0)<<"GB";
        
        if(need_size*8>free_byte)
        {
            size_t max_trick_depth = log2(free_byte/(cut[device_id].cut_points_val.size()*8));
            LOG(INFO)<<"support max trick depth is "<<max_trick_depth;
            shards[device_id].columns.max_trick_depth = max_trick_depth;
            //half of the max nodes
            //shards[device_id].columns.max_trick_nodes = 1<<(max_trick_depth-1);
            //need_size = (1<<max_trick_depth)* cut[device_id].cut_points_val.size(); 
            shards[device_id].columns.max_trick_nodes = free_byte/(cut[device_id].cut_points_val.size()*8)/2;
            need_size = shards[device_id].columns.max_trick_nodes*2* cut[device_id].cut_points_val.size(); 
        }
        LOG(INFO)<<"real last hist size is "<<need_size*8/(1024*1024*1024.0)<<"GB";
        last_hist[device_id].resize(need_size);
   });
    get_bin_ids();
    for (int i = 0; i < param.n_device; ++i) {
        v_columns[i].release();
    }
    SyncMem::clear_cache();
    int gpu_num;
    cudaError_t err = cudaGetDeviceCount(&gpu_num);
    std::atexit([](){
        SyncMem::clear_cache();
    });
}

//new func for update tree
void HistTreeBuilder_single::update_tree() {
    TIMED_FUNC(timerObj);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        auto& sp = this->sp[device_id];
        auto& tree = this->trees[device_id];
        auto sp_data = sp.device_data();
        LOG(DEBUG) << sp;
        int n_nodes_in_level = sp.size();

        Tree::TreeNode *nodes_data = tree.nodes.device_data();
        float_type rt_eps = param.rt_eps;
        float_type lambda = param.lambda;

        device_loop(n_nodes_in_level, [=]__device__(int i) {
            float_type best_split_gain = sp_data[i].gain;
            if (best_split_gain > rt_eps) {
                //do split
                if (sp_data[i].nid == -1) return;
                int nid = sp_data[i].nid;
                Tree::TreeNode &node = nodes_data[nid];
                node.gain = best_split_gain;

                Tree::TreeNode &lch = nodes_data[node.lch_index];//left child
                Tree::TreeNode &rch = nodes_data[node.rch_index];//right child
                lch.is_valid = true;
                rch.is_valid = true;
                node.split_feature_id = sp_data[i].split_fea_id;
                GHPair p_missing_gh = sp_data[i].fea_missing_gh;
                //todo process begin
                node.split_value = sp_data[i].fval;
                node.split_bid = sp_data[i].split_bid;

                lch.sum_gh_pair = sp_data[i].lch_sum_gh;
                if (!sp_data[i].default_right) {
                    lch.sum_gh_pair = lch.sum_gh_pair + p_missing_gh;
                    node.default_right = false;
                }
                else{
                    node.default_right = true;
                }
                rch.sum_gh_pair = node.sum_gh_pair - lch.sum_gh_pair;
                lch.calc_weight(lambda);
                rch.calc_weight(lambda);
            } else {
                //set leaf
                if (sp_data[i].nid == -1) return;
                int nid = sp_data[i].nid;
                Tree::TreeNode &node = nodes_data[nid];
                node.is_leaf = true;
                nodes_data[node.lch_index].is_valid = false;
                nodes_data[node.rch_index].is_valid = false;
            }
        });
        LOG(DEBUG) << tree.nodes;
    });
}



//new func for ins2node update
//new ins2node update
void HistTreeBuilder_single::update_ins2node_id() {
    TIMED_FUNC(timerObj);
    DO_ON_MULTI_DEVICES(param.n_device, [&](int device_id){
        
        auto &columns = shards[device_id].columns;
        HistCut &cut = this->cut[device_id];
        
        auto &csr_row_ptr = this->csr_row_ptr[device_id];
        auto &csr_bin_id  = this->csr_bin_id[device_id];
        
        auto csr_row_ptr_data = csr_row_ptr.device_data();
        auto csr_bin_id_data = csr_bin_id.device_data();

        auto cut_row_ptr = cut.cut_row_ptr.device_data();

        using namespace thrust;
        int n_column = columns.n_column;
        //int nnz = columns.nnz;
        
        // int n_block = fminf((nnz / n_column - 1) / 256 + 1, 4 * 56);

        SyncArray<bool> has_splittable(1);
        auto nid_data = ins2node_id[device_id].device_data();
        const Tree::TreeNode *nodes_data = trees[device_id].nodes.device_data();
        has_splittable.host_data()[0] = false;
        bool *h_s_data = has_splittable.device_data();
        int column_offset = columns.column_offset;
        //auto max_num_bin = param.max_num_bin;

        //两种思路
        //1.得到划分的split bin_id，然后从csr_bin_id中寻找，这样寻找的范围是一个instance中nnz的长度
        //2、得到划分的split feature 和split feature，在csr中寻找instance的feature和对应的value，这需要保留value数组
        //选择第一种
        auto binary_search = [=]__device__( size_t search_begin,  size_t search_end, 
                                            const size_t cut_begin, const size_t cut_end,
                                            const int *csr_bin_id_data) {
            int previous_middle = -1;
            while (search_begin != search_end) {
                int middle = search_begin + (search_end - search_begin)/2;

                if(middle == previous_middle){
                    break;
                }
                previous_middle = middle;
                auto tmp_bin_id = csr_bin_id_data[middle];

                if(tmp_bin_id >= cut_begin && tmp_bin_id < cut_end){
                    return tmp_bin_id;
                }
                else if (tmp_bin_id < cut_begin){
                    search_begin = middle;
                }
                else{
                    search_end = middle;
                }
            }
            //missing values
            return -1;
        };

        //auto loop_search = [=]__device__( size_t search_begin,  size_t search_end, 
        //                                    const size_t cut_begin, const size_t cut_end,
        //                                    const int *csr_bin_id_data) {
        //    for(int i =search_begin;i<=search_end;i++){
        //        auto bin_id = csr_bin_id_data[i];
        //        if(bin_id >= cut_begin && bin_id <cut_end){
        //            return bin_id;
        //        }
        //    }
        //    return -1;
        //};
        //update instance to node map
        device_loop(n_instances, [=]__device__(int iid) {
            int nid = nid_data[iid];
            const Tree::TreeNode &node = nodes_data[nid];
            int split_fid = node.split_feature_id;
            if (node.splittable() && ((split_fid - column_offset < n_column) && (split_fid >= column_offset))) {
                h_s_data[0] = true;
                int split_bid = (int)node.split_bid+cut_row_ptr[split_fid]; 
                int bid = binary_search(csr_row_ptr_data[iid],csr_row_ptr_data[iid+1],
                                        cut_row_ptr[split_fid],cut_row_ptr[split_fid+1],
                                        csr_bin_id_data);
                bool to_left = true;
                //if(bid == -1 ){
                //    to_left = !node.default_right;
                //}
                //else{
                //    if(split_bid == cut_row_ptr[split_fid]){
                //        to_left = false;
                //    }
                //    else if(bid > split_bid){
                //        to_left = false;
                //    }
                //}

                if ((bid == -1 && node.default_right) || (bid >= split_bid && bid>=0))
                    to_left = false;
                 
                if (to_left) {
                    //goes to left child
                    nid_data[iid] = node.lch_index;
                } else {
                    //right child
                    nid_data[iid] = node.rch_index;
                }
            }
        });
        
       
        LOG(DEBUG) << "new tree_id = " << ins2node_id[device_id];
        has_split[device_id] = has_splittable.host_data()[0];
    });
}
