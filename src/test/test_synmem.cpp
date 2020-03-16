#include "gtest/gtest.h"
#include "thundergbm/syncmem.h"


TEST(SyncmemTest, test_constructor) {
    SyncMem smem1(10);
    EXPECT_NE(smem1.host_data(), nullptr);
    EXPECT_EQ(smem1.size(), 10);
    EXPECT_EQ(smem1.head(), SyncMem::HEAD::HOST);

    SyncMem smem2(10);
    EXPECT_EQ(smem2.size(), 10);
    smem2.to_host();
    EXPECT_EQ(smem2.size(), 10) << "The size of smem2 is " << smem2.size();
}


TEST(SyncmemTest, test_set_host_data) {
    SyncMem smem1(10);
    SyncMem smem2(10);
    // copy the data from smem1 to smem2
    smem2.set_host_data(smem1.host_data());
    EXPECT_EQ(smem2.size(), 10);
}

TEST(SyncmemTest, test_set_device_data) {
    SyncMem smem1(20);
    SyncMem smem2(20);
    // copy the data from smem1 to smem2
    smem2.set_device_data(smem2.device_data());
    EXPECT_EQ(smem2.size(), 20);
    // the head flag of smem2 should become DEVICE
    EXPECT_EQ(smem2.head(), SyncMem::HEAD::DEVICE);
}

TEST(SyncmemTest, test_host_to_device) {
    SyncMem smem1(sizeof(int) * 10);
    int *data = static_cast<int *>(smem1.host_data());
    for(int i = 0; i < 10; i++)
        data[i] = i;
    smem1.to_device();
    // the head flag of smem1 should become DEVICE
    EXPECT_EQ(smem1.head(), SyncMem::HEAD::DEVICE);

    // change the data on the host
    for(int i = 0; i < 10; i++)
        data[i] = -10;
    smem1.to_host();
    // the head flag of smem1 should become HOST
    EXPECT_EQ(smem1.head(), SyncMem::HEAD::HOST);

    // reset the data and check if the data has been changed
    data = static_cast<int *>(smem1.host_data());
    for(int i = 0; i < 10; i++)
        EXPECT_EQ(data[i], i);
}

TEST(SyncMem, test_get_device_id) {
    SyncMem smem1(10);
    smem1.to_device();
    EXPECT_EQ(smem1.get_owner_id(), 0) << "the default device id is 0";
}

TEST(SyncMem, test_clear_cache) {
    SyncMem smem(sizeof(int) * 10);
    int *data = static_cast<int *>(smem.host_data());
    for(int i = 0; i < 10; i++)
        data[i] = i;
    for(int i = 0; i < 10; i++)
        EXPECT_EQ(data[i], i);
    smem.clear_cache();
    data = static_cast<int *>(smem.host_data());
    for(int i = 0; i < 10; i++)
        EXPECT_EQ(data[i], i);
}
