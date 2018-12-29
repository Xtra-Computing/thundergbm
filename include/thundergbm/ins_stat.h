//
// Created by shijiashuai on 5/7/18.
//

#ifndef THUNDERGBM_INS_STAT_H
#define THUNDERGBM_INS_STAT_H


#include "syncarray.h"
struct GHPair {
    float_type g;
    float_type h;

    HOST_DEVICE GHPair operator+(const GHPair &rhs) const {
        GHPair res;
        res.g = this->g + rhs.g;
        res.h = this->h + rhs.h;
        return res;
    }

    HOST_DEVICE const GHPair operator-(const GHPair &rhs) const {
        GHPair res;
        res.g = this->g - rhs.g;
        res.h = this->h - rhs.h;
        return res;
    }

    HOST_DEVICE bool operator==(const GHPair &rhs) const {
        return this->g == rhs.g && this->h == rhs.h;
    }

    HOST_DEVICE bool operator!=(const GHPair &rhs) const {
        return !(*this == rhs);
    }

    HOST_DEVICE GHPair() : g(0), h(0) {};

    HOST_DEVICE GHPair(float_type v) : g(v), h(v) {};

    HOST_DEVICE GHPair(float_type g, float_type h) : g(g), h(h) {};

    friend std::ostream &operator<<(std::ostream &os,
                                    const GHPair &p) {
        os << string_format("%f/%f", p.g, p.h);
        return os;
    }
};

class InsStat {
public:

    ///gradient and hessian
    SyncArray<GHPair> gh_pair;
    ///node id
    SyncArray<int> nid;
    ///target value
    SyncArray<float_type> y;
    ///predict value
    SyncArray<float_type> y_predict;

    int n_instances;

    GHPair sum_gh;

    InsStat() = default;

    explicit InsStat(size_t n_instances) {
        resize(n_instances);
    }

    void resize(size_t n_instances);

    void updateGH(bool bagging);

    void do_bagging();
};

#endif //THUNDERGBM_INS_STAT_H
