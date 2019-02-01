//
// Created by qinbin on 2018/5/9.
//
#ifndef THUNDERGBM_QUANTILE_SKETCH_H
#define THUNDERGBM_QUANTILE_SKETCH_H

#include "common.h"
#include <utility>
#include <tuple>

using std::pair;
using std::tuple;
using std::vector;


class entry{
public:
    float_type val;//a cut point candidate
    float_type rmin;//total weights of feature values less than val
    float_type rmax;//total weights of feature values less than or equal to val
    float_type w;
    entry() {};
    entry(float_type val, float_type rmin, float_type rmax, float_type w) : val(val), rmin(rmin), rmax(rmax), w(w) {};
};

class summary{
public:
    int entry_size;
    int entry_reserve_size;
    vector<entry> entries;
    summary(): entry_size(0),entry_reserve_size(0) {
        //entries.clear();
    };
    summary(int entry_size, int reserve_size): entry_size(entry_size), entry_reserve_size(reserve_size) {entries.resize(reserve_size);};
    void Reserve(int size);//reserve memory for the summary
    void Prune(summary& src,int size);//reduce the number of cut point candidates of the summary
    void Merge(summary& src1, summary& src2);//merge two summaries
    void Copy(summary& src);

};

/**
 * @brief: store the <fvalue, weight> pairs before constructing a summary
 */
class Qitem{
public:
    int tail;
    vector<pair<float_type, float_type>> data;
    Qitem(): tail(0) {
        //data.clear();
    };
    void GetSummary(summary& ret);
};


class quanSketch{
public:
    int numOfLevel;//the summary has multiple levels
    int summarySize;//max size of the first level summary
    Qitem Qentry;
    vector<summary> summaries;
    summary t_summary; //for process_nodes
    void Init(int maxn, float_type eps);
    void Add(float_type, float_type);
    void GetSummary(summary& dest);
    quanSketch(): numOfLevel(0), summarySize(0) {
        //summaries.clear();
    };

};
#endif //THUNDERGBM_QUANTILE_SKETCH_H
