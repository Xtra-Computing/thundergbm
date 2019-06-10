ThunderGBM How To
======
This page is for key instructions of intalling, using and contributing to ThunderGBM. Everyone in the community can contribute to ThunderGBM to make it better.

## How to install ThunderGBM
First of all, you need to install the prerequisite libraries and tools. Then you can download and install ThunderGBM. 
### Prerequisites
* cmake 2.8 or above
    * gcc 4.8 or above for Linux | [CUDA](https://developer.nvidia.com/cuda-downloads) 8 or above
    * Visual C++ for Windows | CUDA 10


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

### Build on Windows
You can build the ThunderGBM library as follows:
```bash
cd thundergbm
mkdir build
cd build
cmake .. -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE -G "Visual Studio 15 2017 Win64"
```
You need to change the Visual Studio version if you are using a different version of Visual Studio. Visual Studio can be downloaded from [this link](https://www.visualstudio.com/vs/). The above commands generate some Visual Studio project files, open the Visual Studio project to build ThunderGBM. Please note that CMake should be 3.4 or above for Windows.

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

## How to build the Python wheel file
You have to ensure the repository is identical to the latest one.
* Clone ThunderGBM repository
```bash
git clone https://github.com/zeyiwen/thundergbm.git
cd thundergbm
#under the directory of thundergbm
git submodule init cub && git submodule update
```
* Build the binary
```base
mkdir build && cd build && cmake .. && make -j
```
* Build the python wheel file
    - change directory to python by `cd ../python`
    - update the version you are going to release in [setup.py](https://github.com/Xtra-Computing/thundergbm/blob/c89d6da6008f945c09aae521c95cfe5b8bdd8db5/python/setup.py#L20)
    - you may need to install the `wheel` dependency by ``pip3 install wheel``
```bash
python3 setup.py bdist_wheel
```
* Upload the wheel file to [Pypi.org](https://pypi.org)
    - you may need to install the `twine` dependency by ``pip3 install twine``
    - you need to use ``python3 -m twine upload dist/*`` if ``twine`` is not included in ``PATH``
```sybase
twine upload dist/* --verbose
```
* [Recommended] Draw a new release on [Release](https://github.com/Xtra-Computing/thundergbm/releases)
    * state the bug fixed or new functions.