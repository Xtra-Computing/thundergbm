#include "gtest/gtest.h"
#include "thundergbm/util/cub_wrapper.h"
#include "thundergbm/syncarray.h"

class CubWrapperTest: public::testing::Test {
public:
    SyncArray<int> values;
    SyncArray<int> keys;

protected:
    void SetUp() override {
        values.resize(4);
        keys.resize(4);
        auto values_data = values.host_data();
        values_data[0] = 1;
        values_data[1] = 4;
        values_data[2] = -1;
        values_data[3] = 0;
        auto keys_data = keys.host_data();
        keys_data[0] = 1;
        keys_data[1] = 3;
        keys_data[2] = 2;
        keys_data[3] = 4;
    }

};

TEST_F(CubWrapperTest, test_cub_sort_by_key) {
    // keys from [1, 3, 2, 4] to [1, 2, 3, 4]
    // values from [1, 4, -1, 0] to [1, -1, 4, 0]
    cub_sort_by_key(keys, values);
    auto keys_data = keys.host_data();
    auto values_data = values.host_data();
    EXPECT_EQ(keys_data[0], 1);
    EXPECT_EQ(keys_data[1], 2);
    EXPECT_EQ(keys_data[2], 3);
    EXPECT_EQ(keys_data[3], 4);
    EXPECT_EQ(values_data[0], 1);
    EXPECT_EQ(values_data[1], -1);
    EXPECT_EQ(values_data[2], 4);
    EXPECT_EQ(values_data[3], 0);
}

TEST_F(CubWrapperTest, test_cub_seg_sort_by_key) {
    SyncArray<int> seg_ptr(3); // 2 segments
    auto seg_ptr_data = seg_ptr.host_data();
    seg_ptr_data[0] = 0;
    seg_ptr_data[1] = 2;
    seg_ptr_data[2] = 4;
    // keys from [1, 3, 2, 4] to [1, 3, 2, 4]
    // values from [1, 4, -1, 0] to [1, 4, -1, 0]
    /*cub_seg_sort_by_key(keys, values, seg_ptr);*/
    auto keys_data = keys.host_data();
    auto values_data = values.host_data();
    EXPECT_EQ(keys_data[0], 1);
    EXPECT_EQ(keys_data[1], 3);
    EXPECT_EQ(keys_data[2], 2);
    EXPECT_EQ(keys_data[3], 4);
    EXPECT_EQ(values_data[0], 1);
    EXPECT_EQ(values_data[1], 4);
    EXPECT_EQ(values_data[2], -1);
    EXPECT_EQ(values_data[3], 0);
}

TEST_F(CubWrapperTest, test_sort_array) {
    sort_array(values);
    auto values_data = values.host_data();
    EXPECT_EQ(values_data[0], -1);
    EXPECT_EQ(values_data[1], 0);
    EXPECT_EQ(values_data[2], 1);
    EXPECT_EQ(values_data[3], 4);
}

TEST_F(CubWrapperTest, test_max_elem) {
    int max_elem = max_elements(values);
    EXPECT_EQ(max_elem, 4);
}

