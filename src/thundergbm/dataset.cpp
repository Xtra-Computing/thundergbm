//
// Created by jiashuai on 18-1-17.
//
#include <omp.h>
#include <thundergbm/dataset.h>
#include <thundergbm/objective/objective_function.h>

#include "thundergbm/dataset.h"

void DataSet::load_from_file(string file_name, GBMParam param) {
    LOG(INFO) << "loading LIBSVM dataset from file \"" << file_name << "\"";
    y.clear();
    csr_row_ptr.resize(1, 0);
    csr_col_idx.clear();
    csr_val.clear();
    label.clear();
    n_features_ = 0;

    std::ifstream ifs(file_name, std::ifstream::binary);
    CHECK(ifs.is_open()) << "file " << file_name << " not found";

    std::array<char, 4 << 20> buffer{};
    const int nthread = omp_get_max_threads();

    auto find_last_line = [](char *ptr, const char *begin) {
        while (ptr != begin && *ptr != '\n' && *ptr != '\r' && *ptr != '\0') --ptr;
        return ptr;
    };

    while (ifs) {
        ifs.read(buffer.data(), buffer.size());
        char *head = buffer.data();
        size_t size = ifs.gcount();
        vector<vector<float_type>> y_(nthread);
        vector<vector<int>> col_idx_(nthread);
        vector<vector<int>> row_len_(nthread);
        vector<vector<float_type>> val_(nthread);

        vector<int> max_feature(nthread, 0);

#pragma omp parallel num_threads(nthread)
        {
            //get working area of this thread
            int tid = omp_get_thread_num();
            size_t nstep = (size + nthread - 1) / nthread;
            size_t sbegin = std::min(tid * nstep, size - 1);
            size_t send = std::min((tid + 1) * nstep, size - 1);
            char *pbegin = find_last_line(head + sbegin, head);
            char *pend = find_last_line(head + send, head);

            //move stream start position to the end of last line
            if (tid == nthread - 1) {
                if (ifs.eof())
                    pend = head + send;
                else
                    ifs.seekg(-(head + send - pend), std::ios_base::cur);
            }

            //read instances line by line
            //TODO optimize parse line
            char *lbegin = pbegin;
            char *lend = lbegin;
            while (lend != pend) {
                //get one line
                lend = lbegin + 1;
                while (lend != pend && *lend != '\n' && *lend != '\r' && *lend != '\0') {
                    ++lend;
                }
                string line(lbegin, lend);
                if (line != "\n") {
                    std::stringstream ss(line);

                    //read label of an instance
                    y_[tid].push_back(0);
                    ss >> y_[tid].back();

                    row_len_[tid].push_back(0);
                    string tuple;
                    while (ss >> tuple) {
                        int i;
                        float v;
                        CHECK_EQ(sscanf(tuple.c_str(), "%d:%f", &i, &v), 2)
                            << "read error, using [index]:[value] format";
//TODO one-based and zero-based
                        col_idx_[tid].push_back(i - 1);//one based
                        val_[tid].push_back(v);
                        if (i > max_feature[tid]) {
                            max_feature[tid] = i;
                        }
                        row_len_[tid].back()++;
                    }
                }
                //read next instance
                lbegin = lend;

            }
        }
        for (int i = 0; i < nthread; i++) {
            if (max_feature[i] > n_features_)
                n_features_ = max_feature[i];
        }
        for (int tid = 0; tid < nthread; tid++) {
            csr_val.insert(csr_val.end(), val_[tid].begin(), val_[tid].end());
            csr_col_idx.insert(csr_col_idx.end(), col_idx_[tid].begin(), col_idx_[tid].end());
            for (int row_len : row_len_[tid]) {
                csr_row_ptr.push_back(csr_row_ptr.back() + row_len);
            }
        }
        for (int i = 0; i < nthread; i++) {
            this->y.insert(y.end(), y_[i].begin(), y_[i].end());
            this->label.insert(label.end(), y_[i].begin(), y_[i].end());
        }
    }
    ifs.close();
    LOG(INFO) << "#instances = " << this->n_instances() << ", #features = " << this->n_features();
    if (ObjectiveFunction::need_load_group_file(param.objective)) load_group_file(file_name + ".group");
    if (ObjectiveFunction::need_group_label(param.objective)) group_label();
}

size_t DataSet::n_features() const {
    return n_features_;
}

size_t DataSet::n_instances() const {
    return this->y.size();
}

void DataSet::load_from_sparse(int n_instances, float *csr_val, int *csr_row_ptr, int *csr_col_idx, float *y) {
    n_features_ = 0;
    this->y.clear();
    this->label.clear();
    this->csr_val.clear();
    this->csr_row_ptr.clear();
    this->csr_col_idx.clear();
    int nnz = csr_row_ptr[n_instances];
    this->y.resize(n_instances);
    this->label.resize(n_instances);
    this->csr_val.resize(nnz);
    this->csr_row_ptr.resize(n_instances + 1);
    this->csr_col_idx.resize(nnz);

    if(sizeof(float_type) == float) {
        memcpy(this->y.data(), y, sizeof(float) * n_instances);
        memcpy(this->csr_val.data(), csr_val, sizeof(float) * nnz);
    }
    else{//move instead of copy for converting float to double
        for(int i = 0; i < n_instances; i++) {
            this->y.data()[i] = y[i];
            this->label.data()[i] = y[i];
        }
        for(int e = 0; e < nnz; e++)
            this->csr_val.data()[e] = csr_val[e];
    }
    memcpy(this->csr_col_idx.data(), csr_col_idx, sizeof(int) * nnz);
    memcpy(this->csr_row_ptr.data(), csr_row_ptr, sizeof(int) * (n_instances + 1));
    for (int i = 0; i < nnz; ++i) {
        if (csr_col_idx[i] > n_features_) n_features_ = csr_col_idx[i];
    }
    n_features_++;//convert from zero-based
    LOG(INFO) << "#instances = " << this->n_instances() << ", #features = " << this->n_features();
}

void DataSet::load_group_file(string file_name) {
    LOG(INFO) << "loading group info from file \"" << file_name << "\"";
    group.clear();
    std::ifstream ifs(file_name, std::ifstream::binary);
    CHECK(ifs.is_open()) << "ranking objective needs a group file, but file " << file_name << " not found";
    int group_size;
    while (ifs >> group_size) group.push_back(group_size);
    LOG(INFO) << "#groups = " << group.size();
    ifs.close();
}

void DataSet::group_label() {
    std::map<float_type, int> label_map;
    label.clear();
    for (int i = 0; i < y.size(); ++i) {
        if(label_map.find(y[i]) == label_map.end()) {
            label_map[y[i]] = label.size();
            label.push_back(y[i]);
        }
        y[i] = label_map[y[i]];
    }
}

