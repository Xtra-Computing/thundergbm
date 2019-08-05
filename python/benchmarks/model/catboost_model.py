from model.base_model import BaseModel
import numpy as np
import catboost as cb
import time
import utils.data_utils as du
from model.datasets import Dataset


class CatboostModel(BaseModel):

    def __init__(self):
        BaseModel.__init__(self)

    def _config_model(self, data):
        self.params['eta'] = self.learning_rate
        self.params['depth'] = self.max_depth
        self.params['l2_leaf_reg'] = self.L2_reg
        self.params['devices'] = "0"
        self.params['max_bin'] = self.max_bin
        self.params['thread_count'] = 20
        self.params['iterations'] = self.num_rounds

        if self.use_gpu:
            self.params['task_type'] = 'GPU'
        else:
            self.params['task_type'] = 'CPU'
        if data.task == "Multiclass classification":
            self.params['loss_function'] = 'MultiClass'
            self.params["classes_count"] = int(np.max(data.y_test) + 1)
            self.params["eval_metric"] = 'MultiClass'
        if data.task == "Classification":
            self.params['loss_function'] = 'Logloss'
        if data.task == "Ranking":
            self.params['loss_function'] = 'YetiRank'
            self.params["eval_metric"] = 'NDCG'

    def _train_model(self, data):
        print(self.params)
        dtrain = cb.Pool(data.X_train, data.y_train)
        if data.task == 'Ranking':
            dtrain.set_group_id(data.groups)

        start = time.time()
        self.model = cb.train(pool=dtrain, params=self.params, )
        elapsed = time.time() - start

        return elapsed


    def _predict(self, data):
        # test dataset for catboost
        cb_test = cb.Pool(data.X_test, data.y_test)
        if data.task == 'Ranking':
            cb_test.set_group_id(data.groups)
        preds = self.model.predict(cb_test)
        metric = self.eval(data, preds)

        return metric

    def model_name(self):
        name = "catboost_"
        use_cpu = "gpu_" if self.use_gpu else "cpu_"
        nr = str(self.num_rounds) + "_"
        return name + use_cpu + nr + str(self.max_depth)


if __name__ == "__main__":
    X, y, groups = du.get_yahoo()
    dataset = Dataset(name='yahoo', task='Ranking', metric='NDCG', get_func=du.get_yahoo)
    print(dataset.X_train.shape)
    print(dataset.y_test.shape)

    t_start = time.time()
    xgbModel = CatboostModel()
    xgbModel.use_gpu = False
    xgbModel.run_model(data=dataset)

    eplased = time.time() - t_start
    print("--------->> " + str(eplased))