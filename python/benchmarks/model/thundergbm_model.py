from model.base_model import BaseModel
import thundergbm as tgb
import time
import numpy as np
import utils.data_utils as du
from model.datasets import Dataset


class ThunderGBMModel(BaseModel):

    def __init__(self, depth=6, n_device=1, n_parallel_trees=1,
                 verbose=0, column_sampling_rate=1.0, bagging=0, tree_method='auto'):
        BaseModel.__init__(self)
        self.verbose = verbose
        self.n_device = n_device
        self.column_sampling_rate = column_sampling_rate
        self.bagging = bagging
        self.n_parallel_trees = n_parallel_trees
        self.tree_method = tree_method
        self.objective = ""
        self.num_class = 1

    def _config_model(self, data):
        if data.task == "Regression":
            self.objective = "reg:linear"
        elif data.task == "Multiclass classification":
            self.objective = "multi:softmax"
            self.num_class = int(np.max(data.y_test) + 1)
        elif data.task == "Classification":
            self.objective = "binary:logistic"
        elif data.task == "Ranking":
            self.objective = "rank:ndcg"
        else:
            raise ValueError("Unknown task: " + data.task)


    def _train_model(self, data):
        if data.task is 'Regression':
            self.model = tgb.TGBMRegressor(tree_method=self.tree_method, depth = self.max_depth, n_trees = 40, n_gpus = 1, \
        min_child_weight = 1.0, lambda_tgbm = 1.0, gamma = 1.0,\
        max_num_bin = 255, verbose = 0, column_sampling_rate = 1.0,\
        bagging = 0, n_parallel_trees = 1, learning_rate = 1.0, \
        objective = "reg:linear", num_class = 1)
        else:
            self.model = tgb.TGBMClassifier(bagging=1, lambda_tgbm=1, learning_rate=0.07, min_child_weight=1.2, n_gpus=1, verbose=0,
                            n_parallel_trees=40, gamma=0.2, depth=self.max_depth, n_trees=40, tree_method=self.tree_method, objective='multi:softprob')
        start = time.time()
        self.model.fit(data.X_train, data.y_train)
        elapsed = time.time() - start
        print("##################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! %.5f" % elapsed)

        return elapsed

    def _predict(self, data):
        pred = self.model.predict(data.X_test)
        metric = self.eval(data, pred)
        return metric


    def model_name(self):
        name = "thundergbm_"
        use_cpu = "gpu_" if self.use_gpu else "cpu_"
        nr = str(self.num_rounds) + "_"
        return name + use_cpu + nr + str(self.max_depth)


if __name__ == "__main__":
    # X, y = du.get_higgs()
    dataset = Dataset(name='higgs', task='Regression', metric='RMSE', get_func=du.get_realsim)
    print(dataset.X_train.shape)
    print(dataset.y_test.shape)

    t_start = time.time()

    tgmModel = ThunderGBMModel()
    tgmModel.tree_method = 'hist'
    tgmModel.run_model(data=dataset)

    eplased = time.time() - t_start

    print("--------->> " + str(eplased))