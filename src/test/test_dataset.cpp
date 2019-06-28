//
// Created by jiashuai on 17-9-17.
//
#include "gtest/gtest.h"
#include "thundergbm/dataset.h"
#include "omp.h"

class DatasetTest : public ::testing::Test {
public:
    GBMParam param;
    vector<float_type> csr_val;
    vector<int> csr_row_ptr;
    vector<int> csr_col_idx;
    vector<float_type> y;
    size_t n_features_;
    vector<float_type> label;
protected:
    void SetUp() override {
        param.depth = 6;
        param.n_trees = 40;
        param.n_device = 1;
        param.min_child_weight = 1;
        param.lambda = 1;
        param.gamma = 1;
        param.rt_eps = 1e-6;
        param.max_num_bin = 255;
        param.verbose = false;
        param.profiling = false;
        param.column_sampling_rate = 1;
        param.bagging = false;
        param.n_parallel_trees = 1;
        param.learning_rate = 1;
        param.objective = "reg:linear";
        param.num_class = 1;
        param.path = "../dataset/test_dataset.txt";
        param.tree_method = "auto";
        if (!param.verbose) {
            el::Loggers::reconfigureAllLoggers(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Info, el::ConfigurationType::Enabled, "True");
        }
        if (!param.profiling) {
            el::Loggers::reconfigureAllLoggers(el::ConfigurationType::PerformanceTracking, "false");
        }
    }

    void load_from_file(string file_name, GBMParam &param) {
        LOG(INFO) << "loading LIBSVM dataset from file \"" << file_name << "\"";
        y.clear();
        csr_row_ptr.resize(1, 0);
        csr_col_idx.clear();
        csr_val.clear();
        n_features_ = 0;

        std::ifstream ifs(file_name, std::ifstream::binary);
        CHECK(ifs.is_open()) << "file " << file_name << " not found";

        int buffer_size = 4 << 20;
        char *buffer = (char *)malloc(buffer_size);
        //array may cause stack overflow in windows
        //std::array<char, 4> buffer{};
        const int nthread = omp_get_max_threads();

        auto find_last_line = [](char *ptr, const char *begin) {
            while (ptr != begin && *ptr != '\n' && *ptr != '\r' && *ptr != '\0') --ptr;
            return ptr;
        };

        while (ifs) {
            ifs.read(buffer, buffer_size);
            char *head = buffer;
            //ifs.read(buffer.data(), buffer.size());
            //char *head = buffer.data();
            size_t size = ifs.gcount();
            vector<vector<float_type>> y_(nthread);
            vector<vector<int>> col_idx_(nthread);
            vector<vector<int>> row_len_(nthread);
            vector<vector<float_type>> val_(nthread);

            vector<int> max_feature(nthread, 0);
            bool is_zeor_base = false;

#pragma omp parallel num_threads(nthread)
            {
                //get working area of this thread
                int tid = omp_get_thread_num();
                size_t nstep = (size + nthread - 1) / nthread;
                size_t sbegin = (std::min)(tid * nstep, size - 1);
                size_t send = (std::min)((tid + 1) * nstep, size - 1);
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
                            if(i - 1 == -1){
                                is_zeor_base = true;
                            }
                            CHECK_GE(i - 1, -1) << "dataset format error";
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
                if(is_zeor_base){
                    for (int i = 0; i < col_idx_[tid].size(); ++i) {
                        col_idx_[tid][i]++;
                    }
                }
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
        free(buffer);
    }
};

TEST_F(DatasetTest, load_dataset){
    DataSet dataset;
    load_from_file(param.path, param);
    dataset.load_from_file(param.path, param);
    printf("### Dataset: %s, num_instances: %d, num_features: %d. ###\n",
           param.path.c_str(),
           dataset.n_instances(),
           dataset.n_features());
    EXPECT_EQ(dataset.n_instances(), 1605);
    EXPECT_EQ(dataset.n_features_, 119);
    EXPECT_EQ(dataset.label[0], -1);
    EXPECT_EQ(dataset.csr_val[1], 1);

    for(int i = 0; i < csr_val.size(); i++)
        EXPECT_EQ(csr_val[i], dataset.csr_val[i]);
    for(int i = 0; i < csr_row_ptr.size(); i++)
        EXPECT_EQ(csr_row_ptr[i], dataset.csr_row_ptr[i]);
    for(int i = 0; i < csr_col_idx.size(); i++)
        EXPECT_EQ(csr_col_idx[i], dataset.csr_col_idx[i]);
}

