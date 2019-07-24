import sys
sys.path.append("../")

import thundergbm
from sklearn.datasets import load_svmlight_file

def load_test_data(file_path):
    X, y = load_svmlight_file(file_path)

    return X, y

def load_groups(file_path):
    groups = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.strip() != '':
                groups.append(int(line.strip()))

    return groups

if __name__ == "__main__":
    X, y = load_test_data('../../dataset/test_dataset.txt')
    groups = load_groups('../../dataset/test_dataset.txt.group')
    # tgbm_model = thundergbm.TGBMClassifier(depth=6, n_trees=40, objective='rank:ndcg')
    tgbm_model = thundergbm.TGBMRanker(depth=6, n_trees=40, objective='rank:ndcg')
    tgbm_model.fit(X, y, groups)
    pred_result = tgbm_model.predict(X, groups)
    print(pred_result)

