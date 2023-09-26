//
// Created by jiashuai on 18-1-17.
//
#include <omp.h>
#include <thundergbm/dataset.h>
#include <thundergbm/objective/objective_function.h>
//#include <array>
#include "thundergbm/dataset.h"
#include "thrust/scan.h"
#include "thrust/execution_policy.h"

size_t DataSet::n_features() const {
    return n_features_;
}

size_t DataSet::n_instances() const {
    return this->y.size();
}

void DataSet::load_from_sparse(int n_instances, float *csr_val, int *csr_row_ptr, int *csr_col_idx, float *y,
        int *group, int num_group, GBMParam &param) {
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

    if (ObjectiveFunction::need_load_group_file(param.objective)) {
        for(int i = 0; i < num_group; i++)
            this->group.emplace_back(group[i]);
        LOG(INFO) << "#groups = " << this->group.size();
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
    LOG(INFO) << group;
    ifs.close();
}

void DataSet::group_label() {
    std::map<float_type, int> label_map;
    std::vector<int> tmp_y(y.begin(),y.end());
    label.clear();

    //get label set
    std::sort(tmp_y.begin(), tmp_y.end());
    auto last = std::unique(tmp_y.begin(), tmp_y.end());
    tmp_y.erase(last, tmp_y.end());
    
    for(int i = 0;i<tmp_y.size();++i){
        label_map[tmp_y[i]] = label.size();
        label.push_back(tmp_y[i]);
    }

    for (int i = 0; i < y.size(); ++i) {
        y[i] = label_map[y[i]];
    }
}

/**
 * return true if a character is related to digit
 */
inline bool isdigitchars(char c) {
    return (c >= '0' && c <= '9') ||
           c == '+' || c == '-' ||
           c == '.' || c == 'e' ||
           c == 'E';
}

/**
 * for converting string to T(int or float)
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
    int temp_vv = atoi(p);
    p = q;
    while (p != end && isblank(*p)) ++p;
    if(p == end || *p != ':') {
        *endptr = p;
        v1 = temp_v;
        return 1;
    }
    v1 = int(temp_vv);
    p++;
    while(p != end && !isdigitchars(*p)) ++p; // begin of next digital string
    q = p;
    while(q != end && isdigitchars(*q)) ++q;  // end of next digital string
    *endptr = q;
    v2 = atof(p);

    return 2;
}

/**
 * skip the comment and blank
 */
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
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

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

    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO) << "Load dataset using time: " << used_time.count() << " s";

    // TODO Estimate the required memory
    int nnz = this->csr_val.size();
    double mem_size = (double)nnz / 1024;
    mem_size /= 1024;
    mem_size /= 1024;
    mem_size *= 12;
    if(mem_size > (5 * param.n_device))
        this->use_cpu = true;
}


char* find_last_line(char *ptr, const char *begin) {
    while(ptr != begin && *ptr != '\n' && *ptr != '\r' && *ptr != '\0') --ptr;
    return ptr;
}


void line_count(const int nthread, const int buffer_size, vector<int> &line_counts, std::ifstream &ifs) {
    char *buffer = (char *)malloc(buffer_size);
    line_counts.emplace_back(0);
    while(ifs) {
        ifs.read(buffer, buffer_size);
        char *head = buffer;
        size_t size = ifs.gcount();
        vector<int> thread_line_counts(nthread);


#pragma omp parallel num_threads(nthread)
        {
            int tid = omp_get_thread_num(); // thread id
            size_t nstep = (size + nthread - 1) / nthread;
            size_t step_begin = (std::min)(tid * nstep, size - 1);
            size_t step_end = (std::min)((tid + 1) * nstep, size - 1);

            // a block is the data partition processed by a thread
            // TODO seems no need to do this
            char *block_begin = find_last_line((head + step_begin), head) + 1;
            char *block_end = find_last_line((head + step_end), block_begin) + 1;

            // move stream start position to the end of the last line after an epoch
            if(tid == nthread - 1) {
                if(ifs.eof()) {
                    block_end = head + step_end;
                } else {
                    ifs.seekg(-(head + step_end - block_end), std::ios_base::cur);
                }
            }
            char *line_begin = block_begin;
            int num_line = 0;
            // to the end of the block
            while(line_begin != block_end) {
                if(*line_begin == '\n')
                    num_line++;
                line_begin++;
            }
            thread_line_counts[tid] = num_line;
        }   // end of multiple threads

        line_counts.insert(line_counts.end(), thread_line_counts.begin(), thread_line_counts.end());
    }  // end while
    free(buffer);
    thrust::inclusive_scan(thrust::host, line_counts.begin(), line_counts.end(), line_counts.begin());
    //line_counts[0] = 0;
    // LOG(INFO) << "Line offset of each block: "<< line_counts;
}


//-----------------------------------------------------------------------------------------------------------
void DataSet::load_csc_from_file(string file_name, GBMParam &param,const int nfeatures) {
    LOG(INFO) << "loading LIBSVM dataset as csc from file ## " << file_name << " ##";
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

    vector<vector<float_type>> feature2values(nfeatures);
    vector<vector<int>> feature2instances(nfeatures);

    // initialize
    y.clear();
    n_features_ = 0;

    // open file stream & get number line of each block
    std::ifstream ifs(file_name, std::ifstream::binary);
    CHECK(ifs.is_open()) << "file ## " << file_name << " ## not found. ";
    const int nthread = omp_get_max_threads();
    int buffer_size = (4 << 20);
    vector<int> line_counts;
    line_count(nthread, buffer_size, line_counts, ifs);
    LOG(DEBUG) << line_counts;

    // reset the file pointer & begin to read data as CSC
    ifs.close();
    ifs.open(file_name, std::ifstream::binary);
    int block_idx = 0;
    char *buffer = (char *)malloc(buffer_size);
    while(ifs) {
        ifs.read(buffer, buffer_size);
        char *head = buffer;
        size_t size = ifs.gcount();
        vector<vector<float_type>> y_(nthread);
        vector<vector<vector<float_type>>> val_(nthread);
        vector<vector<vector<int>>> row_id_(nthread);
        vector<int> max_feature(nthread, 0);
        bool is_zero_base = false;

#pragma omp parallel num_threads(nthread)
        {
            int tid = omp_get_thread_num(); // thread id
            int rid = line_counts[block_idx * nthread + tid];  // line offset in one thread
//            LOG(INFO) << rid;
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

            // TODO consider other methods?
            val_[tid].resize(nfeatures + 1);
            row_id_[tid].resize(nfeatures + 1);
            // to the end of the block
            while(line_begin != block_end) {
                line_end = line_begin + 1;
                while(line_end != block_end && *line_end != '\n' && *line_end != '\r' && *line_end != '\0') ++line_end;
                const char *p = line_begin;
                const char *q = NULL;

                // parse instance label
                float_type label;
                float_type temp_;
                std::ptrdiff_t advanced = ignore_comment_and_blank(p, line_end);
                p += advanced;
                int r = parse_pair<float_type, float_type>(p, line_end, &q, label, temp_);
                if (r < 1) {
                    line_begin = line_end;
                    continue;
                }
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
                        val_[tid][feature_id - 1].push_back(value);
                        row_id_[tid][feature_id - 1].push_back(rid);
                        if(feature_id > max_feature[tid])
                            max_feature[tid] = feature_id;
                    }
                    p = q;
                } // end while
                line_begin = line_end;
                rid++;
            } // end of while(line_begin != block_end)
        } // end of multiple threads
        block_idx++;

        // merge local thread data
        for(int thread_num_feature: max_feature)
            if(thread_num_feature > n_features_)
                n_features_ = thread_num_feature;

        for(int tid = 0; tid < nthread; tid++) {
            for(int fid = 0; fid < n_features_; fid++) {
                if(val_[tid][fid].size() != 0){
                    feature2values[fid].insert(feature2values[fid].end(), val_[tid][fid].begin(), val_[tid][fid].end());
                    feature2instances[fid].insert(feature2instances[fid].end(), row_id_[tid][fid].begin(), row_id_[tid][fid].end());
                }
            }
            y.insert(y.end(), y_[tid].begin(), y_[tid].end());
            label.insert(label.end(), y_[tid].begin(), y_[tid].end());
        }
    } // end of while(ifs)

    ifs.close();
    free(buffer);

    // merge again
    csc_col_ptr.emplace_back(0);
    for(int fid = 0; fid < n_features_; fid++) {
        csc_val.insert(csc_val.end(), feature2values[fid].begin(), feature2values[fid].end());
//        LOG(INFO) << feature2instances[fid];
        csc_row_idx.insert(csc_row_idx.end(), feature2instances[fid].begin(), feature2instances[fid].end());
        csc_col_ptr.emplace_back(feature2values[fid].size());
    }

    LOG(INFO) << "csc_col_ptr: " << csc_col_ptr;
    thrust::inclusive_scan(thrust::host, csc_col_ptr.begin(), csc_col_ptr.end(), csc_col_ptr.begin());

    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO) << "Load dataset using time: " << used_time.count() << " s";

    LOG(INFO) << "csc_col_ptr: " << csc_col_ptr;
    LOG(INFO) << "#features: " << n_features_;
}
