[![Documentation Status](https://readthedocs.org/projects/thundergbm/badge/?version=latest)](https://thundergbm.readthedocs.org)
[![GitHub license](https://img.shields.io/badge/license-apache2-yellowgreen)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/xtra-computing/thundergbm.svg)](https://github.com/xtra-computing/thundergbm/issues)
[![PyPI version](https://badge.fury.io/py/thundergbm.svg)](https://badge.fury.io/py/thundergbm)

<div align="center">
<img src="https://github.com/zeyiwen/thundergbm/blob/master/docs/_static/tgbm-logo.png" width="240" height="200" align=left/>
<img src="https://github.com/zeyiwen/thundergbm/blob/master/docs/_static/lang-logo-tgbm.png" width="270" height="200" align=left/>
<img src="https://github.com/zeyiwen/thundergbm/blob/master/docs/_static/overall.png" width="250" height="200" align=left/>
</div>

[Documentations](docs/index.md) | [Installation](docs/how-to.md#how-to-install-thundergbm) | [Parameters](docs/parameters.md) | [Python (scikit-learn) interface](python/README.md)

## Overview
The mission of ThunderGBM is to help users easily and efficiently apply GBDTs and Random Forests to solve problems. ThunderGBM exploits GPUs to achieve high efficiency. Key features of ThunderGBM are as follows.
* Often by 10x times over other libraries.
* Support Python (scikit-learn) interfaces.
* Supported Operating System(s): Linux and Windows.
* Support classification, regression and ranking.

**Why accelerate GBDT and Random Forests**: A [survey](https://www.kaggle.com/amberthomas/kaggle-2017-survey-results) conducted by Kaggle in 2017 shows that 50%, 46% and 24% of the data mining and machine learning practitioners are users of Decision Trees, Random Forests and GBMs, respectively. 


GBDTs and Random Forests are often used for creating state-of-the-art data science solutions. We've listed three winning solutions using GBDTs below. Please check out the [XGBoost website](https://github.com/dmlc/xgboost/blob/master/demo/README.md#machine-learning-challenge-winning-solutions) for more winning solutions and use cases. Here are some example successes of GDBTs and Random Forests:

- Halla Yang, 2nd place, [Recruit Coupon Purchase Prediction Challenge](https://www.kaggle.com/c/coupon-purchase-prediction), [Kaggle interview](http://blog.kaggle.com/2015/10/21/recruit-coupon-purchase-winners-interview-2nd-place-halla-yang/).
- Owen Zhang, 1st place, [Avito Context Ad Clicks competition](https://www.kaggle.com/c/avito-context-ad-clicks), [Kaggle interview](http://blog.kaggle.com/2015/08/26/avito-winners-interview-1st-place-owen-zhang/).
- Keiichi Kuroyanagi, 2nd place, [Airbnb New User Bookings](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings), [Kaggle interview](http://blog.kaggle.com/2016/03/17/airbnb-new-user-bookings-winners-interview-2nd-place-keiichi-kuroyanagi-keiku/).

## Getting Started

### Prerequisites
* cmake 2.8 or above 
    * gcc 4.8 or above for Linux | [CUDA](https://developer.nvidia.com/cuda-downloads) 9 or above
    * Visual C++ for Windows | CUDA 10

### Quick Install
* For Linux with CUDA 9.0
    * `pip install thundergbm`
    
* For Windows (64bit)
    - Download the Python wheel file (for Python3 or above)
    
        * [CUDA 10.0 - Win64](https://github.com/Xtra-Computing/thundergbm/blob/master/python/dist/thundergbm-0.3.12-py3-none-win_amd64.whl)

    - Install the Python wheel file
    
        * `pip install thundergbm-0.3.4-py3-none-win_amd64.whl`
* Currently only support python3
* After you have installed thundergbm, you can import and use the classifier (similarly for regressor) by:
```python
from thundergbm import TGBMClassifier
clf = TGBMClassifier()
clf.fit(x, y)
```
### Build from source
```bash
git clone https://github.com/zeyiwen/thundergbm.git
cd thundergbm
#under the directory of thundergbm
git submodule init cub && git submodule update
```
### Build on Linux (build instructions for [Windows](docs/how-to.md#build-on-windows))
```bash
#under the directory of thundergbm
mkdir build && cd build && cmake .. && make -j
```

### Quick Start
```bash
./bin/thundergbm-train ../dataset/machine.conf
./bin/thundergbm-predict ../dataset/machine.conf
```
You will see `RMSE = 0.489562` after successful running.

MacOS is not supported, as Apple has [suspended support](https://www.forbes.com/sites/marcochiappetta/2018/12/11/apple-turns-its-back-on-customers-and-nvidia-with-macos-mojave/#5b8d3c7137e9) for some NVIDIA GPUs. We will consider supporting MacOS based on our user community feedbacks. Please stay tuned.

## How to cite ThunderGBM
If you use ThunderGBM in your paper, please cite our work ([preprint](https://github.com/Xtra-Computing/thundergbm/blob/master/thundergbm-full.pdf)).
```
@article{wenthundergbm19,
 author = {Wen, Zeyi and Shi, Jiashuai and He, Bingsheng and Li, Qinbin and Chen, Jian},
 title = {{ThunderGBM}: Fast {GBDTs} and Random Forests on {GPUs}},
 journal = {To appear in arXiv},
 year = {2019}
}
```
### Other related paper
* Zeyi Wen, Bingsheng He, Kotagiri Ramamohanarao, Shengliang Lu, and Jiashuai Shi. Efficient Gradient Boosted Decision Tree Training on GPUs. The 32nd IEEE International Parallel and Distributed Processing Symposium (IPDPS), pages 234-243, 2018. [pdf](https://www.comp.nus.edu.sg/~hebs/pub/IPDPS18-GPUGBDT.pdf)

## Key members of ThunderGBM
* [Zeyi Wen](https://zeyiwen.github.io), NUS (now at The University of Western Australia)
* Jiashuai Shi, SCUT (a visiting student at NUS)
* Qinbin Li, NUS
* Advisor: [Bingsheng He](https://www.comp.nus.edu.sg/~hebs/), NUS
* Collaborators: Jian Chen (SCUT),  Kotagiri Ramamohanarao (The University of Melbourne)

## Other information
* This work is supported by a MoE AcRF Tier 2 grant (MOE2017-T2-1-122) and an NUS startup grant in Singapore.

## Related libraries
* [ThunderSVM](https://github.com/Xtra-Computing/thundersvm), which is another *Thunder* serier software tool developed by our group.
* [XGBoost](https://github.com/dmlc/xgboost) | [LightGBM](https://github.com/Microsoft/LightGBM) | [CatBoost](https://github.com/catboost/catboost) | [cuML](https://github.com/rapidsai/cuml)
