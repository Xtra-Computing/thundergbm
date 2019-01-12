We provide a scikit-learn wrapper interface. Before you use the Python interface, you must build ThunderSVM.

## Instructions for building ThunderGBM
* Please refer to [Installation](http://thundergbm.readthedocs.io/en/latest/how-to.html) for building ThunderGBM.

### Example

* Step 1: go to the Python interface.
```bash
# in thundersvm root directory
cd python
```
* Step 2: create a file called ```tgbm_test.py``` which has the following content.
```python
from thundergbm_scikit import *
from sklearn.datasets import *

x,y = load_svmlight_file("../dataset/test_dataset.txt")
clf = TGBMModel()
clf.fit(x,y)

x2,y2=load_svmlight_file("../dataset/test_dataset.txt")
y_predict=clf.predict(x2, y2)
```
* Step 3: run the python script.
```bash
python tgbm_test.py
```