#include "thundergbm/updater/hist_updater.h"
#include "thundergbm/util/cub_wrapper.h"

void HistUpdater::init_cut(const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats, int n_instances) {
    LOG(TRACE) << "init cut";
    if (!do_cut) {
        v_cut.resize(n_devices);
        for (int i = 0; i < n_devices; i++)
            v_cut[i].get_cut_points(*v_columns[i], stats, max_num_bin, n_instances, i);
        bin_id.resize(n_devices);
        DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
            LOG(TRACE) << string_format("finding split on device %d", device_id);
            get_bin_ids(*v_columns[device_id]);
        });
    }
//    LOG(INFO) << v_cut[0].cut_points;
//    LOG(INFO) << v_cut[0].cut_points_val;
//    LOG(INFO) << v_cut[0].cut_row_ptr;
//    LOG(INFO) << v_columns[0]->csc_val;
//    LOG(INFO) << *bin_id[0];
    do_cut = 1;
}

void HistUpdater::get_bin_ids(const SparseColumns &columns) {
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
    device_loop(n_column, [=]__device__(int cid) {
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
                             const HistCut &cut,
                             SyncArray<SplitPoint> &sp) {
    int n_max_nodes_in_level = static_cast<int>(pow(2, level));
    int nid_offset = static_cast<int>(pow(2, level) - 1);
    int n_column = columns.n_column;
    int n_partition = n_column * n_max_nodes_in_level;
    int nnz = columns.nnz;
    int n_bins = cut.cut_points.size();
    int n_block = std::min((n_bins / n_column - 1) / 256 + 1, 32 * 56);

    LOG(TRACE) << "start finding split";

    //find the best split locally
    {
        using namespace thrust;

        //calculate split information for each split
        int n_split;
        SyncArray<GHPair> gh_prefix_sum(n_bins);
        SyncArray<GHPair> missing_gh(n_partition);
        SyncArray<int_float> rle_key(n_bins);
        auto rle_pid_data = make_transform_iterator(rle_key.device_data(),
                                                    [=]__device__(int_float key) { return get<0>(key); });
        auto rle_fval_data = make_transform_iterator(rle_key.device_data(),
                                                     [=]__device__(int_float key) { return get<1>(key); });
        {
            SyncArray<int> fvid2pid(nnz);
            {
                TIMED_SCOPE(timerObj, "fvid2pid");
                //input
                const int *nid_data = stats.nid.device_data();
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
                            else pid = col_id * n_max_nodes_in_level + nid - nid_offset;
                            fvid2pid_data[fvid] = pid;
                        },
                        n_block);
                LOG(DEBUG) << "fvid2pid " << fvid2pid;
            }

            //gather g/h pairs and do prefix sum
            {
                //get feature value id mapping for partition, new -> old
                SyncArray<int> fvid_new2old(nnz);
                {
                    TIMED_SCOPE(timerObj, "fvid_new2old");
                    sequence(cuda::par, fvid_new2old.device_data(), fvid_new2old.device_end(), 0);

                    //using prefix sum memory for temporary storage
                    cub_sort_by_key(fvid2pid, fvid_new2old, -1, true, (void *) gh_prefix_sum.device_data());
                    LOG(DEBUG) << "sorted fvid2pid " << fvid2pid;
                    LOG(DEBUG) << "fvid_new2old " << fvid_new2old;
                }

                //do prefix sum
                {
                    TIMED_SCOPE(timerObj, "do prefix sum");
                    //same feature value in the same part has the same key.
                    auto key_iter = make_zip_iterator(
                            make_tuple(
                                    fvid2pid.device_data(),
                                    make_permutation_iterator(
//                                            columns.csc_val.device_data(),
                                            bin_id[0]->device_data(),
                                            fvid_new2old.device_data())));//use fvid_new2old to access csc_val
                    n_split = reduce_by_key(
                            cuda::par,
                            key_iter, key_iter + nnz,
                            make_permutation_iterator(                   //ins id -> gh pair
                                    stats.gh_pair.device_data(),
                                    make_permutation_iterator(                 //old fvid -> ins id
                                            columns.csc_row_idx.device_data(),
                                            fvid_new2old.device_data())),             //new fvid -> old fvid
                            rle_key.device_data(),
                            gh_prefix_sum.device_data()
                    ).first - rle_key.device_data();
                    CHECK_LE(n_split, rle_key.size());
                    LOG(INFO) << "RLE ratio = " << (float) n_split / nnz;

                    //prefix sum
                    inclusive_scan_by_key(
                            cuda::par,
                            rle_pid_data, rle_pid_data + n_split,
                            gh_prefix_sum.device_data(),
                            gh_prefix_sum.device_data());
                    LOG(DEBUG) << "gh prefix sum = " << gh_prefix_sum;
                }
            }
        }

        //calculate missing value for each partition
        {
            TIMED_SCOPE(timerObj, "calculate missing value");
            SyncArray<int> pid_ptr(n_partition + 1);
            counting_iterator<int> search_begin(0);
            upper_bound(cuda::par, rle_pid_data, rle_pid_data + n_split, search_begin,
                        search_begin + n_partition, pid_ptr.device_data() + 1);
            LOG(DEBUG) << "pid_ptr = " << pid_ptr;

            auto pid_ptr_data = pid_ptr.device_data();
            auto rle_key_data = rle_key.device_data();
            float_type rt_eps = this->rt_eps;
            device_loop(n_split, [=]__device__(int i) {
                int pid = rle_pid_data[i];
                if (pid == INT_MAX) return;
                float_type f = rle_fval_data[i];
                if ((pid_ptr_data[pid + 1] - 1) == i)//the last RLE
                    //using "get" to get a modifiable lvalue
                    get<1>(rle_key_data[i]) = (f - fabsf(rle_fval_data[pid_ptr_data[pid]]) - rt_eps);
                else
                    //FIXME read/write collision
                    get<1>(rle_key_data[i]) = (f + rle_fval_data[i + 1]) * 0.5f;
            });

            const auto gh_prefix_sum_data = gh_prefix_sum.device_data();
            const auto node_data = tree.nodes.device_data();
            auto missing_gh_data = missing_gh.device_data();
            device_loop(n_partition, [=]__device__(int pid) {
                int nid = pid % n_max_nodes_in_level + nid_offset;
                if (pid_ptr_data[pid + 1] != pid_ptr_data[pid])
                    missing_gh_data[pid] =
                            node_data[nid].sum_gh_pair - gh_prefix_sum_data[pid_ptr_data[pid + 1] - 1];
            });
//                        LOG(DEBUG) << "missing gh = " << missing_gh;
        }

        //calculate gain of each split
        SyncArray<float_type> gain(nnz);
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

            const Tree::TreeNode *nodes_data = tree.nodes.device_data();
            GHPair *gh_prefix_sum_data = gh_prefix_sum.device_data();
            float_type *gain_data = gain.device_data();
            const auto missing_gh_data = missing_gh.device_data();
            //for lambda expression
            float_type mcw = min_child_weight;
            float_type l = lambda;
            device_loop(n_split, [=]__device__(int i) {
                int pid = rle_pid_data[i];
                int nid0 = pid % n_max_nodes_in_level;
                int nid = nid0 + nid_offset;
                if (pid == INT_MAX) return;
                GHPair father_gh = nodes_data[nid].sum_gh_pair;
                GHPair p_missing_gh = missing_gh_data[pid];
                GHPair rch_gh = gh_prefix_sum_data[i];
                float_type max_gain = max(0., compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l));
                if (p_missing_gh.h > 1) {
                    rch_gh = rch_gh + p_missing_gh;
                    float_type temp_gain = compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l);
                    if (temp_gain > 0 && temp_gain - max_gain > 0.1) {//FIXME 0.1?
                        max_gain = -temp_gain;//negative means default split to right
                    }
                }
                gain_data[i] = max_gain;
            });
            LOG(DEBUG) << "gain = " << gain;
        }

        //get best gain and the index of best gain for each feature and each node
        SyncArray<int_float> best_idx_gain(n_partition);
        int n_nodes_in_level;
        {
            TIMED_SCOPE(timerObj, "get best gain");
            auto arg_abs_max = []__device__(const int_float &a, const int_float &b) {
                if (fabsf(get<1>(a)) == fabsf(get<1>(b)))
                    return get<0>(a) < get<0>(b) ? a : b;
                else
                    return fabsf(get<1>(a)) > fabsf(get<1>(b)) ? a : b;
            };

            //reduce to get best split of each node for this feature
            SyncArray<int> feature_nodes_pid(n_partition);
            int n_feature_with_nodes = reduce_by_key(
                    cuda::par,
                    rle_pid_data, rle_pid_data + n_split,
                    make_zip_iterator(make_tuple(counting_iterator<int>(0), gain.device_data())),
                    feature_nodes_pid.device_data(),
                    best_idx_gain.device_data(),
                    thrust::equal_to<int>(),
                    arg_abs_max).second - best_idx_gain.device_data();

            LOG(DEBUG) << "aaa = " << n_feature_with_nodes;
            LOG(DEBUG) << "f n pid" << feature_nodes_pid;
            LOG(DEBUG) << "best idx & gain = " << best_idx_gain;

            auto feature_nodes_pid_data = feature_nodes_pid.device_data();
            device_loop(n_feature_with_nodes, [=]__device__(int i) {
                feature_nodes_pid_data[i] = feature_nodes_pid_data[i] % n_max_nodes_in_level;
            });
            LOG(DEBUG) << "f n pid" << feature_nodes_pid;
            cub_sort_by_key(feature_nodes_pid, best_idx_gain, n_feature_with_nodes);
            LOG(DEBUG) << "f n pid" << feature_nodes_pid;
            LOG(DEBUG) << "best idx & gain = " << best_idx_gain;
            n_nodes_in_level = reduce_by_key(
                    cuda::par,
                    feature_nodes_pid.device_data(), feature_nodes_pid.device_data() + n_feature_with_nodes,
                    best_idx_gain.device_data(),
                    make_discard_iterator(),
                    best_idx_gain.device_data(),
                    thrust::equal_to<int>(),
                    arg_abs_max
            ).second - best_idx_gain.device_data();
            LOG(DEBUG) << "#nodes in level = " << n_nodes_in_level;
            LOG(DEBUG) << "best idx & gain = " << best_idx_gain;
        }

        //get split points
        const int_float *best_idx_gain_data = best_idx_gain.device_data();
        GHPair *gh_prefix_sum_data = gh_prefix_sum.device_data();
        const auto missing_gh_data = missing_gh.device_data();

        sp.resize(n_nodes_in_level);
        auto sp_data = sp.device_data();

        int column_offset = columns.column_offset;
        device_loop(n_nodes_in_level, [=]__device__(int i) {
            int_float bst = best_idx_gain_data[i];
            float_type best_split_gain = get<1>(bst);
            int split_index = get<0>(bst);
            int pid = rle_pid_data[split_index];
            sp_data[i].split_fea_id = (pid == INT_MAX) ? -1 : (pid / n_max_nodes_in_level) + column_offset;
            sp_data[i].nid = (pid == INT_MAX) ? -1 : (pid % n_max_nodes_in_level + nid_offset);
            sp_data[i].gain = fabsf(best_split_gain);
            if (pid != INT_MAX) {//avoid split_index out of bound
                sp_data[i].fval = rle_fval_data[split_index];
                sp_data[i].fea_missing_gh = missing_gh_data[pid];
                sp_data[i].default_right = best_split_gain < 0;
                sp_data[i].rch_sum_gh = gh_prefix_sum_data[split_index];
            }
        });
    }

    LOG(DEBUG) << "split points (gain/fea_id/nid): " << sp;
}

void HistUpdater::grow(Tree &tree, const vector<std::shared_ptr<SparseColumns>> &v_columns, InsStat &stats) {
    TIMED_SCOPE(timerObj, "grow tree");

    int n_instances = stats.n_instances;
    int cur_device = 0;
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    int n_executor;
//    MPI_Comm_size(MPI_COMM_WORLD, &n_executor);

    LOG(TRACE) << "broadcast tree and stats";
    v_stats.resize(n_devices);
    v_trees.resize(n_devices);
    init_tree(tree, stats);
    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
        //copy stats and tree from host (stats, tree) to multi-device (v_stats, v_trees)

        //stats
        int n_instances = stats.n_instances;
        v_stats[device_id].reset(new InsStat());
        InsStat &gpu_stats = *v_stats[device_id];
        gpu_stats.resize(n_instances);
        gpu_stats.gh_pair.copy_from(stats.gh_pair.host_data(), n_instances);
        gpu_stats.nid.copy_from(stats.nid.host_data(), n_instances);
        //        gpu_stats.y.copy_from(stats.y.host_data(), n_instances);
//        gpu_stats.y_predict.copy_from(stats.y_predict.host_data(), n_instances);

        //tree
        v_trees[device_id].reset(new Tree());
        Tree &gpu_tree = *v_trees[device_id];
        gpu_tree.nodes.resize(tree.nodes.size());
        gpu_tree.nodes.copy_from(tree.nodes.host_data(), tree.nodes.size());
    });

    for (int i = 0; i < depth; ++i) {
        LOG(TRACE) << "growing tree at depth " << i;
        vector<SyncArray<SplitPoint>> local_sp(n_devices);
        {
            TIMED_SCOPE(timerObj, "find split");
            DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
                LOG(TRACE) << string_format("finding split on device %d", device_id);
                find_split(i, *v_columns[device_id], *v_trees[device_id], *v_stats[device_id], v_cut[device_id], local_sp[device_id]);
            });
        }

        int n_max_nodes_in_level = 1 << i;//2^i
        int nid_offset = (1 << i) - 1;//2^i - 1
        SyncArray<SplitPoint> global_sp(n_max_nodes_in_level);
        {
            TIMED_SCOPE(timerObj, "split point all reduce");
            if (n_devices > 1)
                split_point_all_reduce(local_sp, global_sp, i);
            else
                global_sp.copy_from(local_sp[0].device_data(), local_sp[0].size());
//            if (n_executor > 1) {
//                if (rank == 0) {
//                    SyncArray<SplitPoint> global_sp2(n_max_nodes_in_level);
//                    MPI_Recv(global_sp2.host_data(), global_sp2.mem_size(), MPI_CHAR, 1, 0, MPI_COMM_WORLD,
//                             MPI_STATUS_IGNORE);
//                    auto global_sp_data = global_sp.host_data();
//                    auto global_sp2_data = global_sp2.host_data();
//                    for (int j = 0; j < global_sp.size(); ++j) {
//                        if (global_sp2_data[j].gain > global_sp_data[j].gain)
//                            global_sp_data[j] = global_sp2_data[j];
//                    }
//                } else if (rank == 1) {
//                    MPI_Send(global_sp.host_data(), global_sp.mem_size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
//                }
//                if (rank == 0) {
//                    MPI_Send(global_sp.host_data(), global_sp.mem_size(), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
//                } else {
//                    MPI_Recv(global_sp.host_data(), global_sp.mem_size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD,
//                             MPI_STATUS_IGNORE);
//                }
//            }
        }
//        LOG(DEBUG) << "rank " << rank << " sp" << global_sp;

        //do split
        {
            TIMED_SCOPE(timerObj, "update tree");
            update_tree(*v_trees[0], global_sp);
        }

        //broadcast tree
        if (n_devices > 1) {
            LOG(TRACE) << "broadcasting updated tree";
            //copy tree on gpu 0 to host, prepare to broadcast
            v_trees[0]->nodes.to_host();
            DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
                v_trees[device_id]->nodes.copy_from(v_trees[0]->nodes.host_data(), v_trees[0]->nodes.size());
            });
        }

        {
            vector<bool> v_has_split(n_devices);
            TIMED_SCOPE(timerObj, "reset ins2node id");
            LOG(TRACE) << "reset ins2node id";
            DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
                v_has_split[device_id] = reset_ins2node_id(*v_stats[device_id], *v_trees[device_id],
                                                           *v_columns[device_id]);
            });

            LOG(TRACE) << "gathering ins2node id";
            //get final result of the reset instance id to node id
//            if (n_executor == 1) {
            bool has_split = false;
            for (int d = 0; d < n_devices; d++) {
                has_split |= v_has_split[d];
            }
            if (!has_split) {
                LOG(INFO) << "no splittable nodes, stop";
                break;
            }
//            } else {
//                todo early stop
//            }
        }

        //get global ins2node id
        {
            TIMED_SCOPE(timerObj, "global ins2node id");
            SyncArray<int> local_ins2node_id(n_instances);
            auto local_ins2node_id_data = local_ins2node_id.device_data();
            auto global_ins2node_id_data = v_stats[0]->nid.device_data();
            for (int d = 1; d < n_devices; d++) {
                CUDA_CHECK(cudaMemcpyPeerAsync(local_ins2node_id_data, cur_device,
                                               v_stats[d]->nid.device_data(), d,
                                               sizeof(int) * n_instances));
                cudaDeviceSynchronize();
                device_loop(n_instances, [=]__device__(int i) {
                    global_ins2node_id_data[i] = (global_ins2node_id_data[i] > local_ins2node_id_data[i]) ?
                                                 global_ins2node_id_data[i] : local_ins2node_id_data[i];
                });
            }
//            if (n_executor > 1) {
//                if (rank == 0) {
//                    MPI_Recv(local_ins2node_id.host_data(), local_ins2node_id.mem_size(), MPI_CHAR, 1, 0,
//                             MPI_COMM_WORLD,
//                             MPI_STATUS_IGNORE);
//                    auto local_ins2node_id_data = local_ins2node_id.device_data();
//                    auto global_ins2node_id_data = stats.nid.device_data();
//                    device_loop(n_instances, [=]__device__(int i) {
//                        global_ins2node_id_data[i] = (global_ins2node_id_data[i] > local_ins2node_id_data[i]) ?
//                                                     global_ins2node_id_data[i] : local_ins2node_id_data[i];
//                    });
//                } else {
//                    MPI_Send(stats.nid.host_data(), stats.nid.mem_size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
//                }
//                if (rank == 0) {
//                    MPI_Send(stats.nid.host_data(), stats.nid.mem_size(), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
//                } else {
//                    MPI_Recv(stats.nid.host_data(), stats.nid.mem_size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//                }
//            }
        }
        //processing missing value
        {
            TIMED_SCOPE(timerObj, "process missing value");
            LOG(TRACE) << "update ins2node id for each missing fval";
            auto global_ins2node_id_data = v_stats[0]->nid.device_data();//essential
            auto nodes_data = v_trees[0]->nodes.device_data();//already broadcast above
            device_loop(n_instances, [=]__device__(int iid) {
                int nid = global_ins2node_id_data[iid];
                //if the instance is not on leaf node and not goes down
                if (nodes_data[nid].splittable() && nid < nid_offset + n_max_nodes_in_level) {
                    //let the instance goes down
                    const Tree::TreeNode &node = nodes_data[nid];
                    if (node.default_right)
                        global_ins2node_id_data[iid] = node.rch_index;
                    else
                        global_ins2node_id_data[iid] = node.lch_index;
                }
            });
            LOG(DEBUG) << "new nid = " << stats.nid;
            //broadcast ins2node id
            v_stats[0]->nid.to_host();
            DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
//                v_stats[device_id]->nid.copy_from(stats.nid.host_data(), stats.nid.size());
                v_stats[device_id]->nid.copy_from(v_stats[0]->nid.host_data(), stats.nid.size());
            });
        }
    }
    tree.nodes.copy_from(v_trees[0]->nodes);
    stats.nid.copy_from(v_stats[0]->nid);
}
