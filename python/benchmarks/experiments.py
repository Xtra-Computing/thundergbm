import pickle
from sklearn.datasets import load_svmlight_file
import os
from sklearn.datasets import dump_svmlight_file


base_dir = './raw_dataset'
plk_dir = './datasets'
for dataset_name in os.listdir(base_dir):
    print(dataset_name)
    # if 'binary' not in dataset_name:
    #     continue
    X, y = load_svmlight_file(os.path.join(base_dir, dataset_name))
    print(X.shape)
    print(y.shape)
    # y2 = [1 if x % 2 == 0 else 0 for x in y]
    # dump_svmlight_file(X, y2, open('SVNH.2classes', 'wb'))
    pickle.dump((X, y), open(os.path.join(plk_dir, dataset_name+'.plk'), 'wb'), protocol=4)
# X, y = load_svmlight_file('datasets/YearPredictMSD.train')