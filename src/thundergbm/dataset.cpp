//
// Created by jiashuai on 18-1-17.
//
#include <omp.h>
#include <thundergbm/dataset.h>
#include <thundergbm/objective/objective_function.h>
//#include <array>
#include "thundergbm/dataset.h"

size_t DataSet::n_features() const {
    return n_features_;
}

size_t DataSet::n_instances() const {
    return this->y.size();
}

void DataSet::load_from_sparse(int n_instances, float *csr_val, int *csr_row_ptr, int *csr_col_idx, float *y, GBMParam &param) {
    n_features_ = 0;
    this->y.clear();
    this->label.clear();
    this->csr_val.clear();
    this->csr_row_ptr.clear();
    this->csr_col_idx.clear();
    int nnz = csr_row_ptr[n_instances];
    this->y.resize(n_instances);
    //this->label.resize(n_instances);
    this->csr_val.resize(nnz);
    this->csr_row_ptr.resize(n_instances + 1);
    this->csr_col_idx.resize(nnz);

    CHECK_EQ(sizeof(float_type), sizeof(float));
//    if(sizeof(float_type) == float) {
//        memcpy(this->y.data(), y, sizeof(float) * n_instances);
//        memcpy(this->csr_val.data(), csr_val, sizeof(float) * nnz);
//    }
//    else{//move instead of copy for converting float to double
//        for(int i = 0; i < n_instances; i++) {
//            this->y.data()[i] = y[i];
//            this->label.data()[i] = y[i];
//        }
//        for(int e = 0; e < nnz; e++)
//            this->csr_val.data()[e] = csr_val[e];
//    }
    if(y != NULL)
        memcpy(this->y.data(), y, sizeof(float) * n_instances);
    memcpy(this->csr_val.data(), csr_val, sizeof(float) * nnz);
    memcpy(this->csr_col_idx.data(), csr_col_idx, sizeof(int) * nnz);
    memcpy(this->csr_row_ptr.data(), csr_row_ptr, sizeof(int) * (n_instances + 1));
    for (int i = 0; i < nnz; ++i) {
        if (csr_col_idx[i] > n_features_) n_features_ = csr_col_idx[i];
    }
    n_features_++;//convert from zero-based
    LOG(INFO) << "#instances = " << this->n_instances() << ", #features = " << this->n_features();

    if (y != NULL && ObjectiveFunction::need_group_label(param.objective)){
        group_label();
        param.num_class = label.size();
    }
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


/**
 * return true if a character is related to digit
 * @param c
 * @return
 */
inline bool isdigitchars(char c) {
    return (c >= '0' && c <= '9') ||
           c == '+' || c == '-' ||
           c == '.' || c == 'e' ||
           c == 'E';
}

/**
 * for
 * @tparam T1
 * @tparam T2
 * @param begin
 * @param end
 * @param endptr
 * @param v1
 * @param v2
 * @return
 */
template<typename T1, typename T2>
inline int parse_pair(const char * begin, const char *end, const char **endptr, T1 &v1, T2 &v2) {
    const char *p = begin;
    // begin of digital string
    while(p != end && !isdigitchars(*p)) ++p;
    if(p == end) {
        *endptr = end;
        return 0;
    }
    const char *q = p;
    // end of digital string
    while(q != end && isdigitchars(*q)) ++q;
    float temp_v = atof(p);
    p = q;
    while (p != end && isblank(*p)) ++p;
    if(p == end || *p != ':') {
        *endptr = p;
        v1 = temp_v;
        return 1;
    }
    v1 = int(temp_v);
    p++;
    while(p != end && !isdigitchars(*p)) ++p; // begin of next digital string
    q = p;
    while(q != end && isdigitchars(*q)) ++q;  // end of next digital string
    *endptr = q;
    v2 = atof(p);

    return 2;
}

template <char kSymbol = '#'>
std::ptrdiff_t ignore_comment_and_blank(char const* beg,
                                        char const* line_end) {
    char const* p = beg;
    std::ptrdiff_t length = std::distance(beg, line_end);
    while (p != line_end) {
        if (*p == kSymbol) {
            // advance to line end, `ParsePair' will return empty line.
            return length;
        }
        if (!isblank(*p)) {
            return std::distance(beg, p);  // advance to p
        }
        p++;
    }
    // advance to line end, `ParsePair' will return empty line.
    return length;
}



void DataSet::load_from_file(string file_name, GBMParam &param) {
    LOG(INFO) << "loading LIBSVM dataset from file ## " << file_name << " ##";

    // initialize
    y.clear();
    csr_val.clear();
    csr_col_idx.clear();
    csr_row_ptr.resize(1, 0);
    n_features_ = 0;

    // open file stream
    std::ifstream ifs(file_name, std::ifstream::binary);
    CHECK(ifs.is_open()) << "file ## " << file_name << " ## not found. ";

    int buffer_size = 4 << 20;
    char *buffer = (char *)malloc(buffer_size);
    const int nthread = omp_get_max_threads();

    auto find_last_line = [](char *ptr, const char *begin) {
        while(ptr != begin && *ptr != '\n' && *ptr != '\r' && *ptr != '\0') --ptr;
        return ptr;
    };

    // read and parse data
    while(ifs) {
        ifs.read(buffer, buffer_size);
        char *head = buffer;
        size_t size = ifs.gcount();

        // create vectors for each thread
        vector<vector<float_type>> y_(nthread);
        vector<vector<float_type>> val_(nthread);
        vector<vector<int>> col_idx(nthread);
        vector<vector<int>> row_len_(nthread);
        vector<int> max_feature(nthread, 0);
        bool is_zero_base = false;

#pragma omp parallel num_threads(nthread)
        {
            int tid = omp_get_thread_num(); // thread id
            size_t nstep = (size + nthread - 1) / nthread;
            size_t step_begin = (std::min)(tid * nstep, size - 1);
            size_t step_end = (std::min)((tid + 1) * nstep, size - 1);

            // a block is the data partition processed by a thread
            char *block_begin = find_last_line((head + step_begin), head);
            char *block_end = find_last_line((head + step_end), block_begin);

            // move stream start position to the end of the last line after an epoch
            if(tid == nthread - 1) {
                if(ifs.eof()) {
                    block_end = head + step_end;
                } else {
                    ifs.seekg(-(head + step_end - block_end), std::ios_base::cur);
                }
            }

            // read instances line by line
            char *line_begin = block_begin;
            char *line_end = line_begin;
            // to the end of the block
            while(line_begin != block_end) {
                line_end = line_begin + 1;
                while(line_end != block_end && *line_end != '\n' && *line_end != '\r' && *line_end != '\0') ++line_end;
                const char *p = line_begin;
                const char *q = NULL;
                row_len_[tid].push_back(0);

                float_type label;
                float_type temp_;
                std::ptrdiff_t advanced = ignore_comment_and_blank(p, line_end);
                p += advanced;
                int r = parse_pair<float_type, float_type>(p, line_end, &q, label, temp_);
                if (r < 1) {
                    line_begin = line_end;
                    continue;
                }
                // parse instance label
                y_[tid].push_back(label);

                // parse feature id and value
                p = q;
                while(p != line_end) {
                    int feature_id;
                    float_type value;
                    std::ptrdiff_t advanced = ignore_comment_and_blank(p, line_end);
                    p += advanced;

                    int r = parse_pair(p, line_end, &q, feature_id, value);
                    if(r < 1) {
                        p = q;
                        continue;
                    }
                    if(r == 2) {
                        col_idx[tid].push_back(feature_id - 1);
                        val_[tid].push_back(value);
                        if(feature_id > max_feature[tid])
                            max_feature[tid] = feature_id;
                        row_len_[tid].back()++;
                    }
                    p = q;
                } // end inner while
                line_begin = line_end;
            } // end outer while
        } // end num_thread
        for (int i = 0; i < nthread; i++) {
            if (max_feature[i] > n_features_)
                n_features_ = max_feature[i];
        }
        for (int tid = 0; tid < nthread; tid++) {
            csr_val.insert(csr_val.end(), val_[tid].begin(), val_[tid].end());
            if(is_zero_base){
                for (int i = 0; i < col_idx[tid].size(); ++i) {
                    col_idx[tid][i]++;
                }
            }
            csr_col_idx.insert(csr_col_idx.end(), col_idx[tid].begin(), col_idx[tid].end());
            for (int row_len : row_len_[tid]) {
                csr_row_ptr.push_back(csr_row_ptr.back() + row_len);
            }
        }
        for (int i = 0; i < nthread; i++) {
            this->y.insert(y.end(), y_[i].begin(), y_[i].end());
            this->label.insert(label.end(), y_[i].begin(), y_[i].end());
        }
    } // end while

    ifs.close();
    free(buffer);
    LOG(INFO) << "#instances = " << this->n_instances() << ", #features = " << this->n_features();
    if (ObjectiveFunction::need_load_group_file(param.objective)) load_group_file(file_name + ".group");
    if (ObjectiveFunction::need_group_label(param.objective)) {
        group_label();
        param.num_class = label.size();
    }
}

