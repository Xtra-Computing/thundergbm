//
// Created by jiashuai on 18-1-17.
//
#include <omp.h>
#include "thundergbm/dataset.h"

void DataSet::load_from_file(string file_name) {
    LOG(INFO) << "loading LIBSVM dataset from file \"" << file_name << "\"";
    y.clear();
    csr_row_ptr.resize(1, 0);
    n_features_ = 0;
    std::ifstream ifs(file_name, std::ifstream::binary);
    CHECK(ifs.is_open()) << "file " << file_name << " not found";

    std::array<char, 4 << 20> buffer{};
    const int nthread = omp_get_max_threads();

    auto find_last_line = [](char *ptr, const char *begin) {
        while (ptr != begin && *ptr != '\n' && *ptr != '\r') --ptr;
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
                y_[tid].push_back(0);
                ss >> y_[tid].back();

                row_len_[tid].push_back(0);
                string tuple;
                while (ss >> tuple) {
                    int i;
                    float v;
                    CHECK_EQ(sscanf(tuple.c_str(), "%d:%f", &i, &v), 2) << "read error, using [index]:[value] format";
//TODO one-based and zero-based
                    col_idx_[tid].push_back(i - 1);//one based
                    val_[tid].push_back(v);
                    if (i > max_feature[tid]) {
                        max_feature[tid] = i;
                    }
                    row_len_[tid].back()++;
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
        }
    }
    LOG(INFO) << "#instances = " << this->n_instances() << ", #features = " << this->n_features();
}

size_t DataSet::n_features() const {
    return n_features_;
}

size_t DataSet::n_instances() const {
    return this->y.size();
}


