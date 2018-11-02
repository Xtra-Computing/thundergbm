#include "thundergbm/updater/hist_updater.h"

void HistUpdater::init_cut(const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats, int n_instances){
    LOG(TRACE)<<"init cut";
    if(!do_cut) {
        v_cut.resize(n_devices);
        for (int i = 0; i < n_devices; i++)
            v_cut[i].get_cut_points(*v_columns[i], stats, max_num_bin, n_instances, i);
        bin_id.resize(n_devices);
        DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
            LOG(TRACE) << string_format("finding split on device %d", device_id);
            get_bin_ids(*v_columns[device_id]);
        });
    }
    do_cut = 1;


}

void HistUpdater::get_bin_ids(const SparseColumns &columns){
    using namespace thrust;
    int cur_device;
    cudaGetDevice(&cur_device);
    int n_column = columns.n_column;
    int nnz = columns.nnz;
    auto cut_row_ptr = v_cut[cur_device].cut_row_ptr.device_data();
    auto cut_points_ptr = v_cut[cur_device].cut_points_val.device_data();
    auto csc_val_data = columns.csc_val.device_data();
    auto csc_col_data = columns.csc_col_ptr.device_data();
    bin_id[cur_device].reset(new SyncArray<int>(nnz));
    auto bin_id_ptr = (*bin_id[cur_device]).device_data();
    device_loop(n_column, [=]__device__(int cid){
        auto cutbegin = cut_points_ptr + cut_row_ptr[cid];
        auto cutend = cut_points_ptr + cut_row_ptr[cid + 1];
        auto valbeign = csc_val_data + csc_col_data[cid];
        auto valend = csc_val_data + csc_col_data[cid + 1];
        lower_bound(cuda::par, cutbegin, cutend, valbeign, valend,
                    bin_id_ptr + csc_col_data[cid], thrust::greater<float_type>());
//        for_each(cuda::par, bin_id_ptr + csc_col_data[cid],
//                 bin_id_ptr + csc_col_data[cid + 1], thrust::placeholders::_1 += cut_row_ptr[cid]);
    });
}
void HistUpdater::find_split(int level, const SparseColumns &columns, const Tree &tree, const InsStat &stats,
                             SyncArray<SplitPoint> &sp) {
    int n_max_nodes_in_level = static_cast<int>(pow(2, level));
    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = columns.n_column;
    int n_partition = n_column * n_max_nodes_in_level;
    int nnz = columns.nnz;
    int n_block = std::min((nnz / n_column - 1) / 256 + 1, 32 * 56);

    int cur_device;
    cudaGetDevice(&cur_device);
    LOG(TRACE) << "start finding split";

    //find the best split locally
    {
        using namespace thrust;
        SyncArray<int> fvid2pid(nnz);

        {
            TIMED_SCOPE(timerObj, "fvid2pid");
            //input
            const int *nid_data = v_stats[cur_device]->nid.device_data();//######### change to vector
            const int *iid_data = columns.csc_row_idx.device_data();

            LOG(TRACE) << "after using v_stats and columns";
            //output
            int *fvid2pid_data = fvid2pid.device_data();
            device_loop_2d(
                    n_column, columns.csc_col_ptr.device_data(),
                    [=]__device__(int col_id, int fvid) {
                        //feature value id -> instance id -> node id
                        int nid = nid_data[iid_data[fvid]];
                        int pid;
                        //if this node is leaf node, move it to the end
                        if (nid < nid_offset) pid = INT_MAX;//todo negative
                        else pid = (nid - nid_offset) * n_column + col_id;
                        fvid2pid_data[fvid] = pid;
                    },
                    n_block);
            cudaDeviceSynchronize();
            LOG(DEBUG) << "fvid2pid " << fvid2pid;
        }


        //gather g/h pairs and do prefix sum
        int n_split;
        SyncArray<GHPair> gh_prefix_sum;
        SyncArray<GHPair> missing_gh(n_partition);
        SyncArray<int> rle_pid;
        SyncArray<float_type> rle_fval;
        //bool do_exact = true;
        {
            //get feature value id mapping for partition, new -> old
            SyncArray<int> fvid_new2old(nnz);
            {
                TIMED_SCOPE(timerObj, "fvid_new2old");
                sequence(cuda::par, fvid_new2old.device_data(), fvid_new2old.device_end(), 0);
                stable_sort_by_key(
                        cuda::par, fvid2pid.device_data(), fvid2pid.device_end(),
                        fvid_new2old.device_data(),
                        thrust::less<int>());
                LOG(DEBUG) << "sorted fvid2pid " << fvid2pid;
                LOG(DEBUG) << "fvid_new2old " << fvid_new2old;
            }
            {
                TIMED_SCOPE(timerObj, "hist");
                n_split = n_partition / n_column * v_cut[cur_device].cut_points.size();
                gh_prefix_sum.resize(n_split);
                rle_pid.resize(n_split);
                rle_fval.resize(n_split);
                int cut_points_size = v_cut[cur_device].cut_points_val.size();

                SyncArray<int> p_size_prefix_sum(n_partition + 1);
                //p_size_prefix_sum.host_data()[0] = 0;
                counting_iterator<int> search_begin(0);
                upper_bound(cuda::par, fvid2pid.device_data(), fvid2pid.device_end(), search_begin,
                            search_begin + n_partition, p_size_prefix_sum.device_data() + 1);
                LOG(TRACE)<<"p_size_prefix_sum:"<<p_size_prefix_sum;
                int n_f = 0;

                cudaMemcpy(&n_f, p_size_prefix_sum.device_data() + n_partition, sizeof(int),
                           cudaMemcpyDeviceToHost);

                auto fval_iter = make_permutation_iterator(columns.csc_val.device_data(),
                                                           fvid_new2old.device_data());


                LOG(TRACE)<<"fvid new2old:"<<fvid_new2old;
                auto iid_iter = make_permutation_iterator(
                        columns.csc_row_idx.device_data(), fvid_new2old.device_data());
                auto p_size_prefix_sum_ptr = p_size_prefix_sum.device_data();

                auto cut_points_ptr = v_cut[cur_device].cut_points_val.device_data();
                auto gh_prefix_sum_ptr = gh_prefix_sum.device_data();

                auto cut_row_ptr = v_cut[cur_device].cut_row_ptr.device_data();
                auto stats_gh_ptr = v_stats[cur_device]->gh_pair.device_data();

                SyncArray<int> cut_off(n_f);
                auto cut_off_ptr = cut_off.device_data();

                auto bin_iter = make_permutation_iterator((*bin_id[cur_device]).device_data(),
                        fvid_new2old.device_data());
                copy(cuda::par, bin_iter, bin_iter + n_f, cut_off_ptr);
                device_loop(n_partition, [=]__device__(int pid){
                    int cut_start = pid / n_column * cut_points_size + cut_row_ptr[pid % n_column];
                    for_each(cuda::par, cut_off_ptr + p_size_prefix_sum_ptr[pid],
                                         cut_off_ptr + p_size_prefix_sum_ptr[pid + 1],
                                         thrust::placeholders::_1 += cut_start);
                });

//                device_loop(n_partition,
//                            [=]__device__(int pid) {
//                                int partition_size = p_size_prefix_sum_ptr[pid + 1] - p_size_prefix_sum_ptr[pid];
//                                auto partition_start = fval_iter + p_size_prefix_sum_ptr[pid];
//                                //auto iid = iid_iter + p_size_prefix_sum_ptr[pid];
//                                auto cbegin = cut_points_ptr + cut_row_ptr[pid % n_column];
//                                auto cend = cut_points_ptr + cut_row_ptr[pid % n_column + 1];
//                                //int bin_size = cend - cbegin;
//                                lower_bound(cuda::par, cbegin, cend, partition_start,
//                                            partition_start + partition_size,
//                                            cut_off_ptr + p_size_prefix_sum_ptr[pid],
//                                            thrust::greater<float_type>());
//                                //replace(cuda::par, cut_off_ptr + p_size_prefix_sum_ptr[pid], cut_off_ptr + p_size_prefix_sum_ptr[pid + 1], bin_size, bin_size - 1);
//
//                                int cut_start = pid / n_column * cut_points_size + cut_row_ptr[pid % n_column];
//                                for_each(cuda::par, cut_off_ptr + p_size_prefix_sum_ptr[pid],
//                                         cut_off_ptr + p_size_prefix_sum_ptr[pid + 1],
//                                         thrust::placeholders::_1 += cut_start);
//
//                            });

                auto gh_insid_ptr = make_permutation_iterator(stats_gh_ptr, iid_iter);
                SyncArray<GHPair> gh_ins(n_f);
                auto gh_ins_ptr = gh_ins.device_data();
                copy(cuda::par, gh_insid_ptr, gh_insid_ptr + n_f, gh_ins_ptr);

                cudaDeviceSynchronize();

                sort_by_key(cuda::par, cut_off_ptr, cut_off_ptr + n_f, gh_ins_ptr);

                SyncArray<int> cut_off_after_reduce(n_split);
                auto cut_off_after_reduce_ptr = cut_off_after_reduce.device_data();

                int n_bin = reduce_by_key(cuda::par, cut_off_ptr, cut_off_ptr + n_f, gh_ins_ptr,
                                          cut_off_after_reduce_ptr, gh_ins_ptr).first -
                            cut_off_after_reduce.device_data();
                LOG(TRACE) << "cut_off_after reduce" << cut_off_after_reduce;

                device_loop(n_bin, [=]__device__(int i) {
                    gh_prefix_sum_ptr[cut_off_after_reduce_ptr[i]].g = gh_ins_ptr[i].g;
                    gh_prefix_sum_ptr[cut_off_after_reduce_ptr[i]].h = gh_ins_ptr[i].h;
                });

                auto rle_fval_ptr = rle_fval.device_data();
                device_loop(n_split, [=]__device__(int i) {
                    rle_fval_ptr[i] = cut_points_ptr[i % cut_points_size];
                });



                auto rle_pid_ptr = rle_pid.device_data();

                device_loop_2d_mod(n_partition, n_column, cut_row_ptr,
                                   [=]__device__(int pid, int cut_off) {
                                       int off = pid / n_column * cut_points_size + cut_off;
                                       rle_pid_ptr[off] = pid;
                                   }, 1);

                inclusive_scan_by_key(
                        cuda::par,
                        rle_pid.device_data(), rle_pid.device_end(),
                        gh_prefix_sum.device_data(),
                        gh_prefix_sum.device_data());


                //const auto gh_prefix_sum_ptr = gh_prefix_sum.device_data();
                const auto node_ptr = tree.nodes.device_data();
                auto missing_gh_ptr = missing_gh.device_data();
                auto cut_row_ptr_device = v_cut[cur_device].cut_row_ptr.device_data();

                device_loop(n_partition, [=]__device__(int pid) {
                    int nid = pid / n_column + nid_offset;
                    int off = pid / n_column * cut_points_size + cut_row_ptr_device[pid % n_column + 1] - 1;
                    if (p_size_prefix_sum_ptr[pid + 1] != p_size_prefix_sum_ptr[pid])
                        missing_gh_ptr[pid] =
                                node_ptr[nid].sum_gh_pair - gh_prefix_sum_ptr[off];
                });

                cudaDeviceSynchronize();
            }
        }

        //calculate gain of each split
        SyncArray<float_type> gain(n_split);
        SyncArray<bool> default_right(n_split);
        {
            TIMED_SCOPE(timerObj, "calculate gain");
            auto compute_gain = []__device__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                             float_type lambda) -> float_type {
                if (lch.h >= min_child_weight && rch.h >= min_child_weight)
                    return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
                           (father.g * father.g) / (father.h + lambda);
                else
                    return 0;
            };

            int *fvid2pid_data = fvid2pid.device_data();
            const Tree::TreeNode *nodes_data = tree.nodes.device_data();
//                float_type *f_val_data = columns.csc_val.device_data();
            GHPair *gh_prefix_sum_data = gh_prefix_sum.device_data();
            float_type *gain_data = gain.device_data();
            bool *default_right_data = default_right.device_data();
            const auto rle_pid_data = rle_pid.device_data();
            const auto missing_gh_data = missing_gh.device_data();
            auto rle_fval_data = rle_fval.device_data();
            //for lambda expression
            float_type mcw = min_child_weight;
            float_type l = lambda;
            device_loop(n_split, [=]__device__(int i) {
                int pid = rle_pid_data[i];
                int nid0 = pid / n_column;
                int nid = nid0 + nid_offset;
                if (pid == INT_MAX) return;
                GHPair father_gh = nodes_data[nid].sum_gh_pair;
                GHPair p_missing_gh = missing_gh_data[pid];
                GHPair rch_gh = gh_prefix_sum_data[i];
                float_type max_gain = compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l);
                if (p_missing_gh.h > 1) {
                    rch_gh = rch_gh + p_missing_gh;
                    float_type temp_gain = compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l);
                    if (temp_gain > 0 && temp_gain - max_gain > 0.1) {
                        max_gain = temp_gain;
                        default_right_data[i] = true;
                    }
                }
                gain_data[i] = max_gain;
            });
            cudaDeviceSynchronize();
            LOG(DEBUG) << "gain = " << gain;
        }

        //get best gain and the index of best gain for each feature and each node
        SyncArray<int_float> best_idx_gain(n_max_nodes_in_level);
        int n_nodes_in_level;
        {
            TIMED_SCOPE(timerObj, "get best gain");
            auto arg_max = []__device__(const int_float &a, const int_float &b) {
                if (get<1>(a) == get<1>(b))
                    return get<0>(a) < get<0>(b) ? a : b;
                else
                    return get<1>(a) > get<1>(b) ? a : b;
            };
            auto in_same_node = [=]__device__(const int a, const int b) {
                return (a / n_column) == (b / n_column);
            };

            //reduce to get best split of each node for this feature
            SyncArray<int> key_test(n_max_nodes_in_level);
            n_nodes_in_level = reduce_by_key(
                    cuda::par,
                    rle_pid.device_data(), rle_pid.device_end(),
                    make_zip_iterator(make_tuple(counting_iterator<int>(0), gain.device_data())),
                    key_test.device_data(),//make_discard_iterator(),
                    best_idx_gain.device_data(),
                    in_same_node,
                    arg_max).second - best_idx_gain.device_data();

            LOG(DEBUG) << "#nodes in level = " << n_nodes_in_level;
            LOG(DEBUG) << "best pid = " << key_test;
            LOG(DEBUG) << "best idx & gain = " << best_idx_gain;
        }

        //get split points
        const int_float *best_idx_gain_data = best_idx_gain.device_data();
        const auto rle_pid_data = rle_pid.device_data();
        //Tree::TreeNode *nodes_data = tree.nodes.device_data();
        GHPair *gh_prefix_sum_data = gh_prefix_sum.device_data();
        const auto rle_fval_data = rle_fval.device_data();
        const auto missing_gh_data = missing_gh.device_data();
        bool *default_right_data = default_right.device_data();

        sp.resize(n_nodes_in_level);
        auto sp_data = sp.device_data();

        int column_offset = columns.column_offset;
        device_loop(n_nodes_in_level, [=]__device__(int i) {
            int_float bst = best_idx_gain_data[i];
            float_type best_split_gain = get<1>(bst);
            int split_index = get<0>(bst);
            int pid = rle_pid_data[split_index];
            sp_data[i].split_fea_id = (pid == INT_MAX) ? -1 : (pid % n_column) + column_offset;
            sp_data[i].nid = (pid == INT_MAX) ? -1 : (pid / n_column + nid_offset);
            sp_data[i].gain = best_split_gain;
            if (pid != INT_MAX) {//avoid split_index out of bound
                sp_data[i].fval = rle_fval_data[split_index];
                sp_data[i].fea_missing_gh = missing_gh_data[pid];
                sp_data[i].default_right = default_right_data[split_index];
                sp_data[i].rch_sum_gh = gh_prefix_sum_data[split_index];
            }
        });
    }

    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}
