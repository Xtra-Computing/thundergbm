//
// Created by qinbin on 2018/5/9.
//
#ifndef THUNDERGBM_QUANTILE_SKETCH_H
#define THUNDERGBM_QUANTILE_SKETCH_H

#include "thundergbm/thundergbm.h"
#include <utility>
#include <tuple>

using std::pair;
using std::tuple;
using std::vector;


class entry{
public:
    float_type val;
    float_type rmin;
    float_type rmax;
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
    void Reserve(int size);
    void Prune(summary& src,int size);
    void Merge(summary& src1, summary& src2);
    void Copy(summary& src);

};

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
    int numOfLevel;
    int summarySize;
    Qitem Qentry;
    vector<summary> summaries;
    summary t_summary; //for temp
    void Init(int maxn, float_type eps);
    void Add(float_type, float_type);
    void GetSummary(summary& dest);
    quanSketch(): numOfLevel(0), summarySize(0) {
        //summaries.clear();
    };

};
#endif //THUNDERGBM_QUANTILE_SKETCH_H
