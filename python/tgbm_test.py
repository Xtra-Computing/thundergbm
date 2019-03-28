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