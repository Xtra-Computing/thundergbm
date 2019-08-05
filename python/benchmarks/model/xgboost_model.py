from model.base_model import BaseModel
import numpy as np
import xgboost as xgb
import time
import utils.data_utils as du
from model.datasets import Dataset


class XGboostModel(BaseModel):

    def __init__(self, use_exact=False, debug_verose=1):
        BaseModel.__init__(self)
        self.use_exact = use_exact
        self.debug_verose = debug_verose

    def _config_model(self, data):
        self.params['max_depth'] = self.max_depth
        self.params['learning_rate'] = self.learning_rate
        self.params['min_split_loss'] = self.min_split_loss
        self.params['min_child_weight'] = self.min_weight
        self.params['alpha'] = self.L1_reg
        self.params['lambda'] = self.L2_reg
        self.params['debug_verbose'] = self.debug_verose
        self.params['max_bin'] = self.max_bin

        if self.use_gpu:
            self.params['tree_method'] = ('gpu_exact' if self.use_exact
                                          else 'gpu_hist')
            self.params['n_gpus'] = 1
        else:
            self.params['nthread'] = 20
            self.params['tree_method'] = ('exact' if self.use_exact else 'hist')

        self.params["predictor"] = "gpu_predictor"
        if data.task == "Regression":
            self.params["objective"] = "reg:squarederror"
        elif data.task == "Multiclass classification":
            self.params["objective"] = "multi:softprob"
            self.params["num_class"] = int(np.max(data.y_test) + 1)
        elif data.task == "Classification":
            self.params["objective"] = "binary:logistic"
        elif data.task == "Ranking":
            self.params["objective"] = "rank:ndcg"
        else:
            raise ValueError("Unknown task: " + data.task)

    def _train_model(self, data):
        print(self.params)
        dtrain = xgb.DMatrix(data.X_train, data.y_train)
        if data.task == 'Ranking':
            dtrain.set_group(data.groups)
        t_start = time.time()
        self.model = xgb.train(self.params, dtrain, self.num_rounds, [(dtrain, "train")])
        elapsed_time = time.time() - t_start

        return elapsed_time

    def _predict(self, data):
        dtest = xgb.DMatrix(data.X_test, data.y_test)
        if data.task == 'Ranking':
            dtest.set_group(data.groups)
        pred = self.model.predict(dtest)
        metric = self.eval(data=data, pred=pred)

        return metric

    def model_name(self):
        name = "xgboost_"
        use_cpu = "gpu_" if self.use_gpu else "cpu_"
        nr = str(self.num_rounds) + "_"
        return name + use_cpu + nr + str(self.max_depth)




if __name__ == "__main__":
    X, y, groups = du.get_yahoo()
    dataset = Dataset(name='yahoo', task='Ranking', metric='NDCG', get_func=du.get_yahoo)
    print(dataset.X_train.shape)
    print(dataset.y_test.shape)

    t_start = time.time()
    xgbModel = XGboostModel()
    xgbModel.use_gpu = False
    xgbModel.run_model(data=dataset)

    eplased = time.time() - t_start
    print("--------->> " + str(eplased))