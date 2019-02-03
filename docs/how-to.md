ThunderGBM How To
======
This page is for key instructions of intalling, using and contributing to ThunderGBM. Everyone in the community can contribute to ThunderGBM to make it better.

## How to install ThunderGBM
First of all, you need to install the prerequisite libraries and tools. Then you can download and install ThunderGBM. 
### Prerequisites
* cmake 2.8 or above | gcc 4.8 or above for Linux | [C++ boost](https://www.boost.org/) | [CUDA](https://developer.nvidia.com/cuda-downloads) 8 or above


### Download
```bash
git clone https://github.com/zeyiwen/thundergbm.git
cd thundergbm
#under the directory of thundergbm
git submodule init cub && git submodule update
```
### Build on Linux 
```bash
cd thundergbm
mkdir build && cd build && cmake .. && make -j
```

### Quick Start
```bash
./bin/thundergbm-train ../dataset/machine.conf
./bin/thundergbm-predict ../dataset/machine.conf
```
You will see `RMSE = 0.489562` after successful running.

## How to build test for ThunderGBM
For building test cases, you also need to obtain ``googletest`` using the following command.
```bash
#under the thundergbm directory
git submodule update --init src/test/googletest
```
After obtaining the ``googletest`` submodule, you can build the test cases by the following commands.
```bash
cd thundergbm
mkdir build && cd build && cmake -DBUILD_TETS=ON .. && make -j
```

## How to use ThunderGBM for ranking

There are two key steps to use ThunderGBM for ranking.
* First, you need to choose ``rank:pairwise`` or ``rank:ndcg`` to set the ``objective`` of ThunderGBM.
* Second, you need to have a file called ``[train_file_name].group`` to specify the number of instances in each query.

The remaining part is the same as classification and regression. Please refer to [Parameters](parameters.md) for more information about setting the parameters.