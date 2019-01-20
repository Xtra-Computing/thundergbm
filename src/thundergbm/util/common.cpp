//
// Created by jiashuai on 18-1-16.
//
#include "thundergbm/common.h"
INITIALIZE_EASYLOGGINGPP

std::ostream &operator<<(std::ostream &os, const int_float &rhs) {
    os << string_format("%d/%f", thrust::get<0>(rhs), thrust::get<1>(rhs));
    return os;
}
