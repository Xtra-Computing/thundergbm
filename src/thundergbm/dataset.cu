//
// Created by jiashuai on 18-1-17.
//
#include <omp.h>
#include <thundergbm/util/cub_wrapper.h>
#include "thundergbm/dataset.h"
#include "cusparse.h"
#include "thrust/sort.h"
#include "thrust/system/cuda/detail/par.h"
#include "thundergbm/util/device_lambda.cuh"

void DataSet::load_from_file(string file_name) {
    LOG(INFO) << "loading LIBSVM dataset from file \"" << file_name << "\"";
    y_.clear();
    features.clear();
    line_num.clear();
    //instances_.clear();
    n_features_ = 0;
    std::ifstream ifs(file_name, std::ifstream::binary);
    CHECK(ifs.is_open()) << "file " << file_name << " not found";

    std::array<char, 2 << 20> buffer{}; //16M
    const int nthread = omp_get_max_threads();

    auto find_last_line = [](char *ptr, const char *begin) {
        while (ptr != begin && *ptr != '\n' && *ptr != '\r') --ptr;
        return ptr;
    };

    string first_line;
    getline(ifs, first_line);
    std::stringstream first_ss(first_line);
    int n_f_first = 0;
    string tuple;
    while(first_ss >> tuple)
        n_f_first++;
    ifs.clear();
    ifs.seekg (0, std::ios::beg);

	int n_sum_line = 0;
    while (ifs) {
        ifs.read(buffer.data(), buffer.size());
        char *head = buffer.data();
        size_t size = ifs.gcount();
        vector<vector<float_type>> y_thread(nthread);
        //vector<node2d> instances_thread(nthread);

        vector<size_t> local_feature(nthread, 0);

		//vector<vector<vector<int>>> index_thread(nthread);
        vector<vector<vector<float_type>>> feature_thread(nthread);
        vector<vector<vector<int>>> line_thread(nthread);
        for(int i = 0; i < nthread; i++){
            feature_thread[i].resize(n_f_first * 2);
            line_thread[i].resize(n_f_first * 2);
        }
        vector<int> n_line(nthread);
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
			int lid = 0;
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

				string tuple;
                //int fid = 0;
                while(ss >> tuple){
                    int i;
                    float v;
                    CHECK_EQ(sscanf(tuple.c_str(), "%d:%f", &i, &v), 2) << "read error, using [index]:[value] format";
                    //index_thread[tid].back().emplace_back(i);
                    if(i > local_feature[tid]){
                        local_feature[tid] = i;
                    }
                    if(i > feature_thread[tid].size()){
                        feature_thread[tid].resize(i);
                        line_thread[tid].resize(i);
                    }

                    feature_thread[tid][i-1].emplace_back(v);
                    line_thread[tid][i-1].emplace_back(lid);
                    //fid++;


                }
                lid++;
                //read next instance
                lbegin = lend;

            }
            n_line[tid] = lid;
        }
        for (int i = 0; i < nthread; i++) {
            if (local_feature[i] > n_features_)
                n_features_ = local_feature[i];
        }
        this->features.resize(n_features_);
        this->line_num.resize(n_features_);
        for(int i = 0; i < nthread; i++) {
            for(int j = 0; j < local_feature[i]; j++) {
                this->features[j].insert(this->features[j].end(),
                                         feature_thread[i][j].begin(),
                                         feature_thread[i][j].end());
                for (int k = 0; k < line_thread[i][j].size(); k++) {
                    line_thread[i][j][k] += n_sum_line;
                }
                this->line_num[j].insert(this->line_num[j].end(),
                                         line_thread[i][j].begin(), line_thread[i][j].end());
            }
            n_sum_line += n_line[i];
        }
		for (int i = 0; i < nthread; i++) {
            this->y_.insert(y_.end(), y_thread[i].begin(), y_thread[i].end());
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
    //return this->instances_.size();
	return this->y_.size();	
}

const vector<float_type> &DataSet::y() const {
    return this->y_;
}

