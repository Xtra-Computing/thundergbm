[![Documentation Status](https://readthedocs.org/projects/thundergbm/badge/?version=latest)](https://thundergbm.readthedocs.org)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/zeyiwen/thundergbm.svg)](https://github.com/zeyiwen/thundergbm/issues)

We are upgrading this project. Please frequently visit this repository for new functionalities.

<div align="center">
<img src="https://github.com/zeyiwen/thundergbm/tree/master/docs/_static/tgbm-logo.png" width="240" height="220" align=left/>
<img src="https://github.com/zeyiwen/thundergbm/tree/master/docs/_static/lang-logo-tgbm.png" width="250" height="200" align=left/>
<img src="https://github.com/zeyiwen/thundergbm/tree/master/docs/_static/overall.png" width="250" height="200" align=left/>
</div>

[Documentations](http://thundergbm.readthedocs.io) | [Parameters](https://thundergbm.readthedocs.io/en/latest/parameters.html) | [Python interface](https://github.com/zeyiwen/thundergbm/tree/master/python)
## Getting Started

### Prerequisites
* cmake 2.8 or above | gcc 4.8 or above for Linux | [CUDA](https://developer.nvidia.com/cuda-downloads) 8 or above

### Download
```bash
git clone https://github.com/zeyiwen/thundergbm.git
git submodule init cub && git submodule update
```
### Build on Linux 
```bash
cd thundergbm
mkdir build && cd build && cmake .. && make -j
```

Build the test cases
```bash
git submodule update --init src/test/googletest
```

### Quick Start
```bash
./bin/thundergbm-train ../dataset/machine.conf
./bin/thundergbm-predict ../dataset/machine.conf
```
You will see `RMSE = 0.489562` after successful running.

### Installation
* Add the require binaries to ```$PATH``` (where ```path_to_cuda``` is the home directory of cuda, 
e.g., ```/usr/local/cuda-9.0/bin/```; ```path_to_mpi``` is the home directory of MPI, e.g., ```/opt/openmpi-gcc/bin/```)
```
export PATH="path_to_cuda:$PATH"
export PATH="path_to_mpi:$PATH"
```
* Build ThunderGBM
```
cd thundergbm
mkdir build && cd build && cmake .. && make -j
```
* Run ThunderGBM with MPI
```
make runtest-mpi
```

## How to cite ThunderGBM
If you use ThunderGBM in your paper, please cite our work.
```
@article{wenthundergbm19,
 author = {Wen, Zeyi and Shi, Jiashuai and He, Bingsheng and Chen, Jian},
 title = {{ThunderGBM}: A Fast Library for {GBDT} and Random Forest Training on {GPUs}},
 journal = {To appear in arXiv},
 year = {2019}
}
```

### Related paper
* Zeyi Wen, Bingsheng He, Kotagiri Ramamohanarao, Shengliang Lu, and Jiashuai Shi. Efficient Gradient Boosted Decision Tree Training on GPUs. The 32nd IEEE International Parallel and Distributed Processing Symposium (IPDPS), pages 234-243, 2018.
