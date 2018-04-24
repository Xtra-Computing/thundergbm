//
// Created by jiashuai on 18-1-18.
//
#include <thundergbm/tree.h>
#include <thundergbm/dataset.h>
#include <thrust/reduce.h>
#include "thrust/adjacent_difference.h"
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include "gtest/gtest.h"
#include "thundergbm/util/device_lambda.cuh"

class UpdaterTest : public ::testing::Test {
public:
    InsStat stats;
    vector<Tree> trees;
    SparseColumns columns;
    unsigned int n_instances;

    int depth = 6;
    int n_trees = 20;
    float_type min_child_weight = 1;
    float_type lambda = 1;
    float_type gamma = 1;
    float_type rt_eps = 1e-6;
    string path = DATASET_DIR "YearPredictionMSD";

    void SetUp() override {
#ifdef NDEBUG
        el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
#endif
        DataSet dataSet;
        dataSet.load_from_file(path);
        n_instances = dataSet.n_instances();
        columns.init(dataSet);
        trees.resize(n_trees);
        stats.init(n_instances);
        int round = 0;
        {
            TIMED_SCOPE(timerObj, "construct tree");
            for (Tree &tree:trees) {
                init_stats(dataSet);
                init_tree(tree);
                int i;
                for (i = 0; i < depth; ++i) {
                    if (!find_split(tree, i)) break;
                }
                //annotate leaf nodes in last level
                {
                    Tree::TreeNode *last_level_nodes_data = tree.nodes.device_data() + int(pow(2, depth) - 1);
                    int n_nodes_last_level = static_cast<int>(pow(2, depth));

                    device_loop(n_nodes_last_level, [=]__device__(int i) {
                        last_level_nodes_data[i].is_leaf = true;
                    });
                }
                LOG(INFO) << string_format("\nbooster[%d]", round) << tree.to_string(depth);
                //compute weights of leaf nodes and predict
                {
                    float_type *y_predict_data = stats.y_predict.device_data();
                    const int *nid_data = stats.nid.device_data();
                    const Tree::TreeNode *nodes_data = tree.nodes.device_data();
                    device_loop(n_instances, [=]__device__(int i) {
                        y_predict_data[i] += nodes_data[nid_data[i]].base_weight;
                    });
                }
                round++;
            }
        }
    }

    void init_stats(DataSet &dataSet) {
        SyncArray<float_type> y(dataSet.y().size());
        y.copy_from(dataSet.y().data(), y.size());
        float_type *y_data = y.device_data();
        GHPair *gh_pair_data = stats.gh_pair.device_data();
        int *nid_data = stats.nid.device_data();
        float_type *stats_y_data = stats.y.device_data();
        float_type *stats_yp_data = stats.y_predict.device_data();
        LOG(DEBUG)<<stats.y_predict;
        LOG(INFO) << "initializing instance statistics";
        device_loop(n_instances, [=] __device__(int i) {
            nid_data[i] = 0;
            stats_y_data[i] = y_data[i];
            gh_pair_data[i].g = stats_yp_data[i] - y_data[i];
            gh_pair_data[i].h = 1;
        });
        LOG(DEBUG) << "gradient/hessian " << stats.gh_pair;
        LOG(DEBUG) << "node id " << stats.nid;
    }

    void init_tree(Tree &tree) {
        tree.init(depth);
        //init root node
        Tree::TreeNode &root_node = tree.nodes.host_data()[0];
        root_node.sum_gh_pair = thrust::reduce(thrust::cuda::par, stats.gh_pair.device_data(),
                                               stats.gh_pair.device_end());
        root_node.is_valid = true;
        root_node.calc_weight(lambda);
        LOG(DEBUG) << root_node.sum_gh_pair;
    }

    //return true if there are splittable nodes
    bool find_split(Tree &tree, int level) {
        int n_max_nodes_in_level = static_cast<int>(pow(2, level));
        int nid_offset = static_cast<int>(pow(2, level) - 1);

        //for each feature, get the best split index for each node
        for (int col_id = 0; col_id < columns.n_column; ++col_id) {
            using namespace thrust;
            int start = columns.csc_col_ptr.host_data()[col_id];
            int nnz = columns.csc_col_ptr.host_data()[col_id + 1] - start;//number of non-zeros

            LOG(DEBUG) << "nnz = " << nnz;
            //get node id for each feature value
            SyncArray<int> f_nid(nnz);
            {
                int *f_nid_data = f_nid.device_data();
                int *nid_data = stats.nid.device_data();
                int *iid_data = columns.csc_row_ind.device_data() + start;

                device_loop(nnz, [=]__device__(int fid) {
                    //feature id -> instance id -> node id
                    int nid = nid_data[iid_data[fid]];
                    //if this node is leaf node, move it to the end
                    if (nid < nid_offset) nid = INT_MAX;
                    f_nid_data[fid] = nid;
                });
                LOG(DEBUG) << "f_nid " << f_nid;
            }

            //get feature id mapping for partition, new -> old
            SyncArray<int> fid_new2old(nnz);
            {
                sequence(cuda::par, fid_new2old.device_data(), fid_new2old.device_end(), 0);
                stable_sort_by_key(cuda::par, f_nid.device_data(), f_nid.device_end(), fid_new2old.device_data(),
                                   thrust::less<int>());
                LOG(DEBUG) << "sorted f_nid " << f_nid;
                LOG(DEBUG) << "fid_new2old " << fid_new2old;
            }

            //gather g/h pairs and do prefix sum
            SyncArray<GHPair> gh_prefix_sum(nnz);
            {
                GHPair *original_gh_data = stats.gh_pair.device_data();
                GHPair *gh_prefix_sum_data = gh_prefix_sum.device_data();
                int *fid_new2old_data = fid_new2old.device_data();
                int *iid_data = columns.csc_row_ind.device_data() + start;

                device_loop(nnz, [=]__device__(int new_fid) {
                    //new local feature id -> old global feature id -> instance id
                    int iid = iid_data[fid_new2old_data[new_fid]];
                    gh_prefix_sum_data[new_fid] = original_gh_data[iid]; //gathering
                });
                LOG(DEBUG) << "gathered g/h " << gh_prefix_sum;

                inclusive_scan_by_key(cuda::par, f_nid.device_data(), f_nid.device_end(),
                                      gh_prefix_sum.device_data(),
                                      gh_prefix_sum.device_data());
                LOG(DEBUG) << "prefix sum " << gh_prefix_sum;
            }

            //find node start position
            SyncArray<int> node_ptr(n_max_nodes_in_level + 1);
            {
                counting_iterator<int> search_begin(nid_offset);

                upper_bound(cuda::par, f_nid.device_data(), f_nid.device_end(), search_begin,
                            search_begin + n_max_nodes_in_level, node_ptr.device_data() + 1);
                LOG(DEBUG) << node_ptr;
            }

            //calculate gain of each split
            SyncArray<float_type> gain(nnz);
            SyncArray<bool> default_right(nnz);
            {
                auto compute_gain = []__device__(GHPair father, GHPair lch, GHPair rch, float_type min_child_weight,
                                                 float_type lambda) -> float_type {
                    if (lch.h > min_child_weight && rch.h > min_child_weight)
                        return (lch.g * lch.g) / (lch.h + lambda) + (rch.g * rch.g) / (rch.h + lambda) -
                               (father.g * father.g) / (father.h + lambda);
                    else
                        return 0;
                };

                int *node_ptr_data = node_ptr.device_data();
                int *f_nid_data = f_nid.device_data();
                Tree::TreeNode *nodes_data = tree.nodes.device_data();
                float_type *f_val_data = columns.csc_val.device_data() + start;
                int *fid_new2old_data = fid_new2old.device_data();
                GHPair *gh_prefix_sum_data = gh_prefix_sum.device_data();
                float_type *gain_data = gain.device_data();
                bool *default_right_data = default_right.device_data();
                //for lambda expression
                float_type mcw = min_child_weight;
                float_type l = lambda;
                float_type rt_eps = this->rt_eps;

                device_loop(nnz, [=]__device__(int fid) {
                    int nid = f_nid_data[fid];//node id
                    if (nid == INT_MAX) return;
                    int begin = node_ptr_data[nid - nid_offset];//feature begin position for this node
                    int end = node_ptr_data[nid - nid_offset + 1] - 1;//feature end position for this node
                    GHPair father_gh = nodes_data[nid].sum_gh_pair;
                    GHPair missing_gh = father_gh - gh_prefix_sum_data[end];
                    if (fid == begin) {
                        gain_data[fid] = compute_gain(father_gh, father_gh - missing_gh, missing_gh, mcw, l);
                        return;
                    }
                    if (fabsf(f_val_data[fid_new2old_data[fid - 1]] - f_val_data[fid_new2old_data[fid]]) >
                        2 * rt_eps) {
                        GHPair rch_gh = gh_prefix_sum_data[fid - 1];
                        float_type max_gain = compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l);
                        if (missing_gh.h > 1) {
                            rch_gh = rch_gh + missing_gh;
                            float_type temp_gain = compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l);
                            if (temp_gain > 0 && temp_gain - max_gain > 0.1) {
                                max_gain = temp_gain;
                                default_right_data[fid] = true;
                            }
                        }
                        gain_data[fid] = max_gain;
                    };
                });
                LOG(DEBUG) << "gain = " << gain;
            }

            //get best gain and the index of best gain for each node
            {
                typedef tuple<int, float_type> arg_max_t;
                auto arg_max = []__device__(const arg_max_t &a, const arg_max_t &b) {
                    return get<1>(a) > get<1>(b) ? a : b;
                };
                device_vector<int> reduced_nid(n_max_nodes_in_level);
                device_vector<arg_max_t> best_split(n_max_nodes_in_level);
                device_ptr<float_type> gain_data = device_pointer_cast(gain.device_data());
                device_ptr<int> f_nid_data = device_pointer_cast(f_nid.device_data());

                //reduce to get best split of each node for this feature
                reduce_by_key(f_nid_data, f_nid_data + nnz,
                              make_zip_iterator(
                                      make_tuple(counting_iterator<int>(0), gain_data)),
                              reduced_nid.begin(), best_split.begin(), equal_to<int>(),
                              arg_max);

                //update global best split for each node
                device_ptr<arg_max_t> best_split_data = best_split.data();
                device_ptr<int> reduced_nid_data = reduced_nid.data();
                const int *node_ptr_data = node_ptr.device_data();
                Tree::TreeNode *nodes_data = tree.nodes.device_data();
                GHPair *gh_prefix_sum_data = gh_prefix_sum.device_data();
                int *fid_new2old_data = fid_new2old.device_data();
                float_type *f_val_data = columns.csc_val.device_data() + start;//offset to this feature
                bool *default_right_data = default_right.device_data();
                float_type rt_eps = this->rt_eps;
                float_type lambda = this->lambda;

                device_loop(reduced_nid.size(), [=]__device__(int i) {
                    int nid = reduced_nid_data[i];
                    if (nid == INT_MAX) return;
                    arg_max_t bst = best_split_data[nid - nid_offset];
                    float_type split_gain = get<1>(bst);
                    int split_index = get<0>(bst);
                    Tree::TreeNode &node = nodes_data[nid];
                    if (split_gain > node.gain) {
                        //save best
                        node.gain = split_gain;
                        node.split_index = fid_new2old_data[split_index];

                        //update best split info
                        Tree::TreeNode &lch = nodes_data[nid * 2 + 1];//left child
                        Tree::TreeNode &rch = nodes_data[nid * 2 + 2];//right child
                        lch.is_valid = true;
                        rch.is_valid = true;
                        node.fid = col_id;
                        int begin = node_ptr_data[nid - nid_offset];
                        int end = node_ptr_data[nid - nid_offset + 1] - 1;
                        GHPair missing_gh = node.sum_gh_pair - gh_prefix_sum_data[end];
                        if (split_index == begin) {
                            node.split_value = f_val_data[fid_new2old_data[end]] -
                                               fabsf(f_val_data[fid_new2old_data[split_index]]) - rt_eps;
                            lch.sum_gh_pair = missing_gh;
                            rch.sum_gh_pair = gh_prefix_sum_data[end];
                        } else {
                            node.split_value = (f_val_data[fid_new2old_data[split_index]] +
                                                f_val_data[fid_new2old_data[split_index - 1]]) * 0.5f;
                            rch.sum_gh_pair = gh_prefix_sum_data[split_index - 1];
                            if (default_right_data[split_index]) {
                                rch.sum_gh_pair = rch.sum_gh_pair + missing_gh;
                                node.default_right = true;
                            }
                            lch.sum_gh_pair = node.sum_gh_pair - rch.sum_gh_pair;
                        }
                        lch.calc_weight(lambda);
                        rch.calc_weight(lambda);
                    }
                });
            }
        }

        //annotate valid nodes
        {
            Tree::TreeNode *nodes_data = tree.nodes.device_data();
            float_type gamma = this->gamma;

            device_loop(n_max_nodes_in_level, [=]__device__(int i) {
                int nid = i + nid_offset;
                Tree::TreeNode &node = nodes_data[nid];
                if (node.gain < gamma) {
                    node.is_leaf = true;
                    nodes_data[nid * 2 + 1].is_valid = false;
                    nodes_data[nid * 2 + 2].is_valid = false;
                }
            });
        }

        SyncArray<bool> has_splittable(1);
        //set new node id for each instance
        {
            int *nid_data = stats.nid.device_data();
            const int *iid_data = columns.csc_row_ind.device_data();
            const Tree::TreeNode *nodes_data = tree.nodes.device_data();
            const int *col_ptr_data = columns.csc_col_ptr.device_data();
            has_splittable.host_data()[0] = false;
            bool *h_s_data = has_splittable.device_data();

            device_lambda_2d_sparse(columns.n_column, col_ptr_data,
                                    [=]__device__(int col_id, int fid) {
                                        //feature id -> instance id
                                        int iid = iid_data[fid];
                                        //instance id -> node id
                                        int nid = nid_data[iid];
                                        //node id -> node
                                        const Tree::TreeNode &node = nodes_data[nid];
                                        //if the node splits on this feature
                                        if (node.splittable() && node.fid == col_id) {
                                            h_s_data[0] = true;
                                            //node goes to next level
                                            nid *= 2;
                                            if (fid - col_ptr_data[col_id] < node.split_index)
                                                //goes to right child
                                                nid += 2;
                                            else
                                                //left child
                                                nid += 1;
                                            nid_data[iid] = nid;
                                        }
                                    });

            //processing missing value
            device_loop(n_instances, [=]__device__(int iid) {
                int nid = nid_data[iid];
                //if the instance is not on leaf node and not goes down
                if (nodes_data[nid].splittable() && nid < nid_offset + n_max_nodes_in_level) {
                    //let the instance goes down
                    nid_data[iid] *= 2;
                    if (nodes_data[nid].default_right)
                        nid_data[iid] += 2;
                    else
                        nid_data[iid] += 1;
                }
            });
        }
        LOG(DEBUG) << "new nid = " << stats.nid;
        return has_splittable.host_data()[0];
    }

};


TEST_F(UpdaterTest, test) {
}