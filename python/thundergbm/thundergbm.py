from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin

import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_X_y

from ctypes import *
from os import path
from sys import platform

dirname = path.dirname(path.abspath(__file__))

if platform == "linux" or platform == "linux2":
    shared_library_name = "libthundergbm.so"
elif platform == "win32":
    shared_library_name = "thundergbm.dll"
elif platform == "darwin":
    shared_library_name = "libthundergbm.dylib"
else:
    raise EnvironmentError("OS not supported!")

if path.exists(path.abspath(path.join(dirname, shared_library_name))):
    lib_path = path.abspath(path.join(dirname, shared_library_name))
else:
    lib_path = path.join(dirname, "../../build/lib", shared_library_name)

if path.exists(lib_path):
    thundergbm = CDLL(lib_path)
else:
    raise RuntimeError("Please build the library first!")

OBJECTIVE_TYPE = ['reg:linear', 'reg:logistic', 'multi:softprob', 'multi:softmax', 'rank:pairwise', 'rank:ndcg']
ThundergbmBase = BaseEstimator
ThundergbmRegressorBase = RegressorMixin
ThundergbmClassifierBase = ClassifierMixin


class TGBMModel(ThundergbmBase):
    def __init__(self, depth, n_trees,
                 n_device, min_child_weight, lambda_tgbm, gamma, max_num_bin,
                 verbose, column_sampling_rate, bagging,
                 n_parallel_trees, learning_rate, objective,
                 num_class, tree_method):
        self.depth = depth
        self.n_trees = n_trees
        self.n_device = n_device
        self.min_child_weight = min_child_weight
        self.lambda_tgbm = lambda_tgbm
        self.gamma = gamma
        self.max_num_bin = max_num_bin
        self.verbose = verbose
        self.column_sampling_rate = column_sampling_rate
        self.bagging = bagging
        self.n_parallel_trees = n_parallel_trees
        self.learning_rate = learning_rate
        self.objective = objective
        self.num_class = num_class
        self.path = path
        self.tree_method = tree_method
        self.model = None
        self.tree_per_iter = -1
        self.group_label = None

    def fit(self, X, y):
        sparse = sp.isspmatrix(X)
        if sparse is False:
            X = sp.csr_matrix(X)
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')

        fit = self._sparse_fit

        fit(X, y)
        return self

    def _sparse_fit(self, X, y):
        X.data = np.asarray(X.data, dtype=np.float32, order='C')
        X.sort_indices()
        data = X.data.ctypes.data_as(POINTER(c_float))
        indices = X.indices.ctypes.data_as(POINTER(c_int32))
        indptr = X.indptr.ctypes.data_as(POINTER(c_int32))
        y = np.asarray(y, dtype=np.float32, order='C')
        label = y.ctypes.data_as(POINTER(c_float))
        group_label = (c_float * len(set(y)))()
        n_class = (c_int * 1)()
        n_class[0] = self.num_class
        tree_per_iter_ptr = (c_int * 1)()
        self.model = (c_long * 1)()
        thundergbm.sparse_train_scikit(X.shape[0], data, indptr, indices, label, self.depth, self.n_trees,
                                       self.n_device, c_float(self.min_child_weight), c_float(self.lambda_tgbm),
                                       c_float(self.gamma),
                                       self.max_num_bin, self.verbose, c_float(self.column_sampling_rate), self.bagging,
                                       self.n_parallel_trees, c_float(self.learning_rate),
                                       self.objective.encode('utf-8'),
                                       n_class, self.tree_method.encode('utf-8'), byref(self.model), tree_per_iter_ptr,
                                       group_label)
        self.num_class = n_class[0]
        self.tree_per_iter = tree_per_iter_ptr[0]
        self.group_label = [group_label[idx] for idx in range(len(set(y)))]
        if self.model is None:
            print("The model returned is empty!")
            exit()

    def predict(self, X):
        if self.model is None:
            print("Please train the model first or load model from file!")
            raise ValueError
        sparse = sp.isspmatrix(X)
        if sparse is False:
            X = sp.csr_matrix(X)
        X.data = np.asarray(X.data, dtype=np.float32, order='C')
        X.sort_indices()
        data = X.data.ctypes.data_as(POINTER(c_float))
        indices = X.indices.ctypes.data_as(POINTER(c_int32))
        indptr = X.indptr.ctypes.data_as(POINTER(c_int32))
        self.predict_label_ptr = (c_float * X.shape[0])()
        if self.group_label is not None:
            group_label = (c_float * len(self.group_label))()
            group_label[:] = self.group_label
        else:
            group_label = None
        thundergbm.sparse_predict_scikit(
            X.shape[0],
            data,
            indptr,
            indices,
            self.predict_label_ptr,
            byref(self.model),
            self.n_trees,
            self.tree_per_iter,
            self.objective.encode('utf-8'),
            self.num_class,
            c_float(self.learning_rate),
            group_label
        )
        predict_label = [self.predict_label_ptr[index] for index in range(0, X.shape[0])]
        self.predict_label = np.asarray(predict_label)
        return self.predict_label

    def save_model(self, model_path):
        if self.model is None:
            print("Please train the model first or load model from file!")
            raise ValueError
        if self.group_label is not None:
            group_label = (c_float * len(self.group_label))()
            group_label[:] = self.group_label
        thundergbm.save(
            model_path.encode('utf-8'),
            self.objective.encode('utf-8'),
            c_float(self.learning_rate),
            self.num_class,
            self.n_trees,
            self.tree_per_iter,
            byref(self.model),
            group_label
        )

    def load_model(self, model_path):
        self.model = (c_long * 1)()
        learning_rate = (c_float * 1)()
        n_class = (c_int * 1)()
        n_trees = (c_int * 1)()
        tree_per_iter = (c_int * 1)()
        thundergbm.load_model(
            model_path.encode('utf-8'),
            learning_rate,
            n_class,
            n_trees,
            tree_per_iter,
            byref(self.model)
        )
        if self.model is None:
            raise ValueError("Model is None.")
        self.learning_rate = learning_rate[0]
        self.num_class = n_class[0]
        self.n_trees = n_trees[0]
        self.tree_per_iter = tree_per_iter[0]
        group_label = (c_float * self.num_class)()
        thundergbm.load_config(
            model_path.encode('utf-8'),
            group_label
        )
        self.group_label = [group_label[idx] for idx in range(self.num_class)]


class TGBMClassifier(TGBMModel, ThundergbmClassifierBase):
    def __init__(self, depth=6, n_trees=40,
                 n_device=1, min_child_weight=1.0, lambda_tgbm=1.0, gamma=1.0, max_num_bin=255,
                 verbose=0, column_sampling_rate=1.0, bagging=0,
                 n_parallel_trees=1, learning_rate=1.0, objective="multi:softmax",
                 num_class=1, tree_method="auto"):
        super().__init__(depth=depth, n_trees=n_trees,
                         n_device=n_device, min_child_weight=min_child_weight, lambda_tgbm=lambda_tgbm, gamma=gamma,
                         max_num_bin=max_num_bin,
                         verbose=verbose, column_sampling_rate=column_sampling_rate, bagging=bagging,
                         n_parallel_trees=n_parallel_trees, learning_rate=learning_rate, objective=objective,
                         num_class=num_class, tree_method=tree_method)


class TGBMRegressor(TGBMModel, ThundergbmRegressorBase):
    def __init__(self, depth=6, n_trees=40,
                 n_device=1, min_child_weight=1.0, lambda_tgbm=1.0, gamma=1.0, max_num_bin=255,
                 verbose=0, column_sampling_rate=1.0, bagging=0,
                 n_parallel_trees=1, learning_rate=1.0, objective="reg:linear",
                 num_class=1, tree_method="auto"):
        super().__init__(depth=depth, n_trees=n_trees,
                         n_device=n_device, min_child_weight=min_child_weight, lambda_tgbm=lambda_tgbm, gamma=gamma,
                         max_num_bin=max_num_bin,
                         verbose=verbose, column_sampling_rate=column_sampling_rate, bagging=bagging,
                         n_parallel_trees=n_parallel_trees, learning_rate=learning_rate, objective=objective,
                         num_class=num_class, tree_method=tree_method)
