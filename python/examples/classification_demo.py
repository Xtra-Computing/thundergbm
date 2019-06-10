import sys
sys.path.append("../")

from thundergbm import TGBMClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    x, y = load_digits(return_X_y=True)
    clf = TGBMClassifier()
    clf.fit(x, y)
    y_pred = clf.predict(x)
    accuracy = accuracy_score(y, y_pred)
    print(accuracy)