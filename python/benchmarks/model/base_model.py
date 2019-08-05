import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


class BaseModel(object):
    """
        Base model to run the test
    """

    def __init__(self):
        self.max_depth = 6
        self.learning_rate = 1
        self.min_split_loss = 1
        self.min_weight = 1
        self.L1_reg = 1
        self.L2_reg = 1
        self.num_rounds = 40
        self.max_bin = 255
        self.use_gpu = True
        self.params = {}

        self.model = None # self.model is different with different libraries

    def _config_model(self, data):
        """
        To config the model with different params
        """
        pass

    def _train_model(self, data):
        """
        To train model
        :param data:
        :return:
        """
        pass

    def _predict(self, data):
        pass

    def eval(self, data, pred):
        """
        To eval the predict results with specified metric
        :param data:
        :param pred:
        :return:
        """
        if data.metric == "RMSE":
            with open('pred', 'w') as f:
                for x in pred:
                    f.write(str(x) + '\n')
            return np.sqrt(mean_squared_error(data.y_test, pred))
        elif data.metric == "Accuracy":
            # Threshold prediction if binary classification
            if data.task == "Classification":
                pred = pred > 0.5
            elif data.task == "Multiclass classification":
                if pred.ndim > 1:
                    pred = np.argmax(pred, axis=1)
            return accuracy_score(data.y_test, pred)
        else:
            raise ValueError("Unknown metric: " + data.metric)

    def run_model(self, data):
        """
        To run model
        :param data:
        :return:
        """
        self._config_model(data)
        elapsed = self._train_model(data)
        # metric = 0
        metric = self._predict(data)
        print("##### Elapsed time: %.5f #####" % (elapsed))
        print("##### Predict %s: %.4f #####" % (data.metric, metric))

        return elapsed, metric

    def model_name(self):
        pass

