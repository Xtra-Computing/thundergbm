from model.base_model import BaseModel
import numpy as np
import lightgbm as lgb
import time
import utils.data_utils as du
from model.datasets import Dataset


class LightGBMModel(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)

    def _config_model(self, data):
        self.params['task'] = 'train'
        self.params['boosting_type'] = 'gbdt'
        self.params['max_depth'] = 6
        self.params['num_leaves'] = 2 ** self.params['max_depth']   # for max_depth is 6
        # self.params['min_sum_hessian+in_leaf'] = 1
        self.params['min_split_gain'] = self.min_split_loss
        self.params['min_child_weight'] = self.min_weight
        self.params['lambda_l1'] = self.L1_reg
        self.params['lambda_l2'] = self.L2_reg
        self.params['max_bin'] = self.max_bin
        self.params['num_threads'] = 20

        if self.use_gpu:
            self.params['device'] = 'gpu'
        else:
            self.params['device'] = 'cpu'
        if data.task == "Regression":
            self.params["objective"] = "regression"
        elif data.task == "Multiclass classification":
            self.params["objective"] = "multiclass"
            self.params["num_class"] = int(np.max(data.y_test) + 1)
        elif data.task == "Classification":
            self.params["objective"] = "binary"
        elif data.task == "Ranking":
            self.params["objective"] = "lambdarank"
        else:
            raise ValueError("Unknown task: " + data.task)


    def _train_model(self, data):
        print(self.params)
        lgb_train = lgb.Dataset(data.X_train, data.y_train)
        if data.task == 'Ranking':
            lgb_train.set_group(data.groups)

        start = time.time()
        self.model = lgb.train(self.params,
                        lgb_train,
                        num_boost_round=self.num_rounds)
        elapsed = time.time() - start

        return elapsed

    def _predict(self, data):
        pred = self.model.predict(data.X_test)
        metric = self.eval(data, pred)

        return metric

    def model_name(self):
        name = "lightgbm_"
        use_cpu = "gpu_" if self.use_gpu else "cpu_"
        nr = str(self.num_rounds) + "_"
        return name + use_cpu + nr + str(self.max_depth)


if __name__ == "__main__":
    X, y, groups = du.get_yahoo()
    dataset = Dataset(name='yahoo', task='Ranking', metric='NDCG', get_func=du.get_yahoo)
    print(dataset.X_train.shape)
    print(dataset.y_test.shape)

    t_start = time.time()
    xgbModel = LightGBMModel()
    xgbModel.use_gpu = False
    xgbModel.run_model(data=dataset)

    eplased = time.time() - t_start
    print("--------->> " + str(eplased))
