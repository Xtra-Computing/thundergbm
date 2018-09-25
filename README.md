### Upgrade
We are upgrading this library, and plan to push the code to GitHub in Sep 2018. Stay tuned!

### Related paper
* Zeyi Wen, Bingsheng He, Kotagiri Ramamohanarao, Shengliang Lu, and Jiashuai Shi. Efficient Gradient Boosted Decision Tree Training on GPUs. The 32nd IEEE International Parallel and Distributed Processing Symposium (IPDPS), pages 234-243, 2018.

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
