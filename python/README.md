We provide a scikit-learn wrapper interface. Before you use the Python interface, you must build ThunderSVM.

## Instructions for building ThunderGBM
* Please refer to [Installation](http://thundergbm.readthedocs.io/en/latest/how-to.html) for building ThunderGBM.

## Prerequisites
* numpy | scipy | sklearn

## Example

* Step 1: go to the Python interface.
```bash
# in thundersvm root directory
cd python
```
* Step 2: create a file called ```tgbm_test.py``` which has the following content.
```python
from thundergbm_scikit import *
from sklearn.datasets import *
from sklearn.metrics import mean_squared_error
from math import sqrt

x,y = load_svmlight_file("../dataset/test_dataset.txt")
clf = TGBMModel()
clf.fit(x,y)

x2,y2=load_svmlight_file("../dataset/test_dataset.txt")
y_predict=clf.predict(x2)

rms = sqrt(mean_squared_error(y2, y_predict))
print(rms)

```
* Step 3: run the python script.
```bash
python tgbm_test.py
```

## ThunderGBM class
*class TGBMModel(depth = 6, num_round = 40, n_device = 1, min_child_weight = 1.0, lambda_tgbm = 1.0, gamma = 1.0, max_num_bin = 255, verbose = 0, column_sampling_rate = 1.0, bagging = 0, n_parallel_trees = 1, learning_rate = 1.0, objective = "reg:linear", num_class = 1, path = "../dataset/test_dataset.txt", out_model_name = "tgbm.model", in_model_name = "tgbm.model"))*

### Parametes
Please refer to [parameter page](https://github.com/zeyiwen/thundergbm/blob/master/docs/parameters.md) in ThunderGBM documentations.