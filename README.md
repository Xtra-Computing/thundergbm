We are upgrading this project. Please frequently visit this repository for new fuctionalities.


## Getting Started

### Prerequisites
* cmake 2.8 or above 
* gcc 4.8 or above for Linux
* [CUDA](https://developer.nvidia.com/cuda-downloads) 8 or above

### Download
```bash
git clone https://github.com/zeyiwen/thundergbm.git
git submodule init cub && git submodule update
```

### Build test on Linux
```bash
git submodule update --init src/test/googletest
```

### Build on Linux 

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

### Related paper
* Zeyi Wen, Bingsheng He, Kotagiri Ramamohanarao, Shengliang Lu, and Jiashuai Shi. Efficient Gradient Boosted Decision Tree Training on GPUs. The 32nd IEEE International Parallel and Distributed Processing Symposium (IPDPS), pages 234-243, 2018.
