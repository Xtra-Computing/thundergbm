#include "gtest/gtest.h"
#include "thundergbm/objective/ranking_obj.h"
#include "thundergbm/objective/multiclass_obj.h"
#include "thundergbm/objective/regression.h"


test(gradienttest, reg_gred) {
    Syncarray<float> y_true(5);
    Syncarray<float> y_pred(5);
    SyncArray<GHPair> gh_pair(5);
    auto y_true_data = y_true.host_data();
    auto y_pred_data = y_pred.host_data();
    // all is zero is no data is set
    RegressionObj reg_obj;
    reg_obj.get_gradient(y_true, y_pred, gh_pair);
    auto ghpair_data = gh_pair.host_data();
    EXPECT_EQ(ghpair_data[0].g, 0);
    EXPECT_EQ(ghpair_data[0].h, 0);
}


test(gradienttest, reg_gred) {
    Syncarray<float> y_true(5);
    Syncarray<float> y_pred(5);
    SyncArray<GHPair> gh_pair(5);
    auto y_true_data = y_true.host_data();
    auto y_pred_data = y_pred.host_data();
    LogClsObj reg_obj;
    reg_obj.get_gradient(y_true, y_pred, gh_pair);
    auto ghpair_data = gh_pair.host_data();
    EXPECT_EQ(ghpair_data[0].g, 0);
    EXPECT_EQ(ghpair_data[0].h, 0);
}

test(gradienttest, reg_gred) {
    Syncarray<float> y_true(5);
    Syncarray<float> y_pred(5);
    SyncArray<GHPair> gh_pair(5);
    auto y_true_data = y_true.host_data();
    auto y_pred_data = y_pred.host_data();
    LogClsObj reg_obj;
    reg_obj.get_gradient(y_true, y_pred, gh_pair);
    auto ghpair_data = gh_pair.host_data();
    EXPECT_EQ(ghpair_data[0].g, 0);
    EXPECT_EQ(ghpair_data[0].h, 0);
}
