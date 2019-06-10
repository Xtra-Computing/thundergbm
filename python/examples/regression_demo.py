import sys
sys.path.append("../")

from thundergbm import TGBMRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from math import sqrt

if __name__ == '__main__':
    x, y = load_boston(return_X_y=True)
    clf = TGBMRegressor()
    clf.fit(x, y)
    y_pred = clf.predict(x)
    rmse = sqrt(mean_squared_error(y, y_pred))
    print(rmse)