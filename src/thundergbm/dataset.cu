//
// Created by jiashuai on 18-1-17.
//
#include <omp.h>
#include "thundergbm/dataset.h"
#include "cusparse.h"
#include "thrust/sort.h"
#include "thrust/system/cuda/detail/par.h"

void DataSet::load_from_file(string file_name) {
    auto find_last_line = [](char *ptr, const char *begin) {
        while (ptr != begin && *ptr != '\n' && *ptr != '\r') --ptr;
        return ptr;
    };
    LOG(INFO) << "loading LIBSVM dataset from file \"" << file_name << "\"";
    y_.clear();
    instances_.clear();
    n_features_ = 0;
    std::ifstream ifs(file_name, std::ifstream::binary);
    CHECK(ifs.is_open()) << "file " << file_name << " not found";

    std::array<char, 2 << 20> buffer{}; //16M
    const int nthread = omp_get_max_threads();

    while (ifs) {
        ifs.read(buffer.data(), buffer.size());
        char *head = buffer.data();
        size_t size = ifs.gcount();
        vector<vector<float_type>> y_thread(nthread);
        vector<node2d> instances_thread(nthread);

        vector<size_t> local_feature(nthread, 0);
#pragma omp parallel num_threads(nthread)
        {
            //get working area of this thread
            int tid = omp_get_thread_num();
            size_t nstep = (size + nthread - 1) / nthread;
            size_t sbegin = std::min(tid * nstep, size);
            size_t send = std::min((tid + 1) * nstep, size);
            char *pbegin = find_last_line(head + sbegin, head);
            char *pend = find_last_line(head + send, head);

            //move stream start position to the end of last line
            if (tid == nthread - 1) ifs.seekg(pend - head - send, std::ios_base::cur);

            //read instances line by line
            char *lbegin = pbegin;
            char *lend = lbegin;
            while (lend != pend) {
                //get one line
                lend = lbegin + 1;
                while (lend != pend && *lend != '\n' && *lend != '\r') {
                    ++lend;
                }
                string line(lbegin, lend);
                std::stringstream ss(line);

                //read label of an instance
                y_thread[tid].emplace_back();
                ss >> y_thread[tid].back();

                //read features of an instance
                instances_thread[tid].emplace_back();
                string tuple;
                while (ss >> tuple) {
                    int i;
                    float_type v;
                    CHECK_EQ(sscanf(tuple.c_str(), "%d:%f", &i, &v), 2) << "read error, using [index]:[value] format";
                    instances_thread[tid].back().emplace_back(i, v);
                    if (i > local_feature[tid]) local_feature[tid] = i;
                };

                //read next instance
                lbegin = lend;
            }
        }
        for (int i = 0; i < nthread; i++) {
            if (local_feature[i] > n_features_)
                n_features_ = local_feature[i];
        }
        for (int i = 0; i < nthread; i++) {
            this->y_.insert(y_.end(), y_thread[i].begin(), y_thread[i].end());
            this->instances_.insert(instances_.end(), instances_thread[i].begin(), instances_thread[i].end());
        }
    }
    LOG(INFO) << "#instances = " << this->n_instances() << ", #features = " << this->n_features();
}

const DataSet::node2d &DataSet::instances() const {
    return this->instances_;
}

size_t DataSet::n_features() const {
    return n_features_;
}

size_t DataSet::n_instances() const {
    return this->instances_.size();
}

const vector<float_type> &DataSet::y() const {
    return this->y_;
}

SparseColumns::SparseColumns(const DataSet &dataset) {
    LOG(INFO) << "constructing sparse columns";
    n_column = dataset.n_features();
    size_t n_instances = dataset.n_instances();
    const DataSet::node2d &instances = dataset.instances();

    /**
     * construct csr matrix, then convert to csc matrix and sort columns by feature values
     */
    vector<float_type> csr_val;
    vector<int> csr_col_ind;//index of each value of all the instances
    vector<int> csr_row_ptr(1, 0);//the start positions of the instances

    LOG(DEBUG) << "converting libsvm sparse rows to csr matrix";
    for (const auto &ins : instances) {//convert libsvm format to csr format
        for (const auto &j : ins) {
            csr_val.push_back(j.value);
            csr_col_ind.push_back(j.index - 1);//libSVM data format is one-based, convert to zero-based
        }
        CHECK_LE(csr_row_ptr.back() + ins.size(), INT_MAX);
        csr_row_ptr.push_back(csr_row_ptr.back() + ins.size());
    }

    nnz = csr_val.size();//number of nonzer
    LOG(INFO)
            << string_format("dataset density = %.2f%% (%d feature values)", (float) nnz / n_instances / n_column * 100,
                             nnz);

    LOG(DEBUG) << "copy csr matrix to GPU";
    //three arrays (on GPU/CPU) for csr representation
    SyncArray<float_type> val;
    SyncArray<int> col_ind;
    SyncArray<int> row_ptr;
    val.resize(csr_val.size());
    col_ind.resize(csr_col_ind.size());
    row_ptr.resize(csr_row_ptr.size());

    //copy data to the three arrays
    val.copy_from(csr_val.data(), val.size());
    col_ind.copy_from(csr_col_ind.data(), col_ind.size());
    row_ptr.copy_from(csr_row_ptr.data(), row_ptr.size());

    LOG(DEBUG) << "converting csr matrix to csc matrix";
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    csc_val.resize(nnz);
    csc_row_ind.resize(nnz);
    csc_col_ptr.resize(n_column + 1);

    cusparseScsr2csc(handle, n_instances, n_column, nnz, val.device_data(), row_ptr.device_data(),
                     col_ind.device_data(), csc_val.device_data(), csc_row_ind.device_data(), csc_col_ptr.device_data(),
                     CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);

    LOG(DEBUG) << "sorting feature values";
    int *col_ptr_data = csc_col_ptr.host_data();
    float_type *val_data = csc_val.device_data();
    int *row_ind_data = csc_row_ind.device_data();
    for (int i = 0; i < n_column; ++i) {
        thrust::sort_by_key(thrust::system::cuda::par, val_data + col_ptr_data[i], val_data + col_ptr_data[i + 1],
                            row_ind_data + col_ptr_data[i], thrust::less<float_type>());
    }
}

SparseColumns::~SparseColumns() {

}
