//
// Created by jiashuai on 18-1-18.
//
#include <thundergbm/tree.h>
#include <thundergbm/dataset.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/detail/par.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include "gtest/gtest.h"
#include "thundergbm/util/device_lambda.cuh"

class UpdaterTest : public ::testing::Test {
public:
    InsStat stats;
    Tree tree;
    SparseColumns columns;
    unsigned int n_instances;
    int depth = 4;

    float_type min_child_weight = 1;
    float_type lambda = 1;
    string path = DATASET_DIR "dataset/" "iris.scale";

    void SetUp() override {
        DataSet dataSet;
        dataSet.load_from_file(path);
        n_instances = dataSet.n_instances();
        columns.init(dataSet);
        init_stats(dataSet);
        init_tree();
        find_split(0);
    }

    void init_stats(DataSet &dataSet) {
        stats.init(n_instances);
        SyncArray<float_type> y(dataSet.y().size());
        y.copy_from(dataSet.y().data(), y.size());
        float_type *y_data = y.device_data();
        GHPair *gh_pair_data = stats.gh_pair.device_data();
        int *nid_data = stats.nid.device_data();
        float_type *stats_y_data = stats.y.device_data();
        float_type *stats_yp_data = stats.y_predict.device_data();
        LOG(DEBUG) << "initializing instance statistics";
        device_loop(n_instances, [=] __device__(int i) {
            gh_pair_data[i].g = -y_data[i];
            gh_pair_data[i].h = 1;
            nid_data[i] = 0;
            stats_y_data[i] = y_data[i];
            stats_yp_data[i] = 0;
        });
        LOG(DEBUG) << "gradient/hessian " << stats.gh_pair;
        LOG(DEBUG) << "node id " << stats.nid;
    }

    void init_tree() {
        tree.init(depth);
        //init root node
        Tree::TreeNode &root_node = tree.nodes.host_data()[0];
        root_node.n_instances = n_instances;
        root_node.sum_gh_pair = thrust::reduce(thrust::cuda::par, stats.gh_pair.device_data(),
                                               stats.gh_pair.device_end());
        LOG(DEBUG) << root_node.sum_gh_pair;
    }

    void find_split(int level) {
        //get segment id ([node id][column id])for each feature value
//        SyncArray<int> nid_f(columns.nnz);
//        int *nid_f_data = nid_f.device_data();
//        int *nid_data = stats.nid.device_data();
//        int *iid_data = columns.csc_row_ind.device_data();
        int n_nodes_in_level = static_cast<int>(pow(2, level));

//        device_lambda_2d_sparse(columns.n_column, columns.csc_col_ptr.device_data(),
//                                [=]__device__(int col_id, int fid) {
//                                    feature id -> instance id -> node id
//                                    nid_f_data[fid] = nid_data[iid_data[fid]];
//                                });

        //for each feature, get the best split index for each node
        SyncArray<float_type> best_gain(n_nodes_in_level);
        SyncArray<int> best_split_index(n_nodes_in_level);
        for (int col_id = 0; col_id < columns.n_column; ++col_id) {
            LOG(DEBUG) << "processing feature " << col_id;
            using namespace thrust;
            int start = columns.csc_col_ptr.host_data()[col_id];
            int nnz = columns.csc_col_ptr.host_data()[col_id + 1] - start;//number of non-zeros

            //get node id for each feature value
            SyncArray<int> f_nid(nnz);
            {
                int *f_nid_data = f_nid.device_data();
                int *nid_data = stats.nid.device_data();
                int *iid_data = columns.csc_row_ind.device_data() + start;
                device_loop(nnz, [=]__device__(int fid) {
                    //feature id -> instance id -> node id
                    f_nid_data[fid] = nid_data[iid_data[fid]];
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

                inclusive_scan_by_key(cuda::par, f_nid.device_data(), f_nid.device_end(), gh_prefix_sum.device_data(),
                                      gh_prefix_sum.device_data());
                LOG(DEBUG) << "prefix sum " << gh_prefix_sum;
            }

            //find node start position
            SyncArray<int> node_ptr(n_nodes_in_level + 1);
            {
                counting_iterator<int> search_begin(0);
                upper_bound(cuda::par, f_nid.device_data(), f_nid.device_end(), search_begin,
                            search_begin + n_nodes_in_level, node_ptr.device_data() + 1);
                LOG(DEBUG) << node_ptr;
            }

            //calculate gain of each split
            SyncArray<float_type> gain(nnz);
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
                //for lambda expression
                float_type mcw = min_child_weight;
                float_type l = lambda;
                device_loop(nnz, [=]__device__(int fid) {
                    int nid = f_nid_data[fid];//node id
                    int begin = node_ptr_data[nid];//feature begin position for this node
                    int end = node_ptr_data[nid + 1];//feature end position for this node
                    GHPair father_gh = nodes_data[nid].sum_gh_pair;
                    GHPair missing_gh = father_gh - gh_prefix_sum_data[end];
                    if (fid == begin) {
                        gain_data[fid] = compute_gain(father_gh, father_gh - missing_gh, missing_gh, mcw, l);
                        return;
                    }
                    if (f_val_data[fid_new2old_data[fid] - 1] != f_val_data[fid_new2old_data[fid]]) {
                        GHPair rch_gh = gh_prefix_sum_data[fid - 1];
                        float_type max_gain = compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l);
//                        if (missing_gh.h < 1) {
//                            gain_data[fid] = max_gain;
//                        }
                        rch_gh = rch_gh + missing_gh;
                        float_type temp_gain = compute_gain(father_gh, father_gh - rch_gh, rch_gh, mcw, l);
                        if (temp_gain > 0 && temp_gain - max_gain > 0.1) {
                            max_gain = temp_gain;
                            //right[fid] = true;
                        }
                        gain_data[fid] = max_gain;
                    };
                });
                LOG(DEBUG) << gain;
            }
        }
    }
};

TEST_F(UpdaterTest, test) {

}