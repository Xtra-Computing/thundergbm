from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import mean_squared_error, accuracy_score

import numpy as np
import scipy.sparse as sp
import statistics

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
# print(lib_path)
if path.exists(lib_path):
    thundergbm = CDLL(lib_path)
else:
    raise RuntimeError("Please build the library first!")

OBJECTIVE_TYPE = ['reg:linear', 'reg:logistic', 'binary:logistic',
                  'multi:softprob', 'multi:softmax', 'rank:pairwise', 'rank:ndcg']

ESTIMATOR_TYPE = ['classifier', 'regressor']

ThundergbmBase = BaseEstimator
ThundergbmRegressorBase = RegressorMixin
ThundergbmClassifierBase = ClassifierMixin


class TGBMModel(ThundergbmBase):
    def __init__(self, depth, n_trees,
                 n_gpus, min_child_weight, lambda_tgbm, gamma, max_num_bin,
                 verbose, column_sampling_rate, bagging,
                 n_parallel_trees, learning_rate, objective,
                 num_class, tree_method):
        self.depth = depth
        self.n_trees = n_trees
        self.n_gpus = n_gpus
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


    def __del__(self):
        if self.model is not None:
            thundergbm.model_free(byref(self.model))

    def _construct_groups(self, groups):
        in_groups = None
        num_groups = 0
        if groups is not None:
            num_groups = len(groups)
            groups = np.asarray(groups, dtype=np.int32, order='C')
            in_groups = groups.ctypes.data_as(POINTER(c_int32))

        return in_groups, num_groups

    def fit(self, X, y, groups=None):
        if self.model is not None:
            thundergbm.model_free(byref(self.model))
            self.model = None
        sparse = sp.isspmatrix(X)
        if sparse is False:
            X = sp.csr_matrix(X)
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')

        fit = self._sparse_fit

        fit(X, y, groups=groups)
        return self

    def _sparse_fit(self, X, y, groups=None):
        X.data = np.asarray(X.data, dtype=np.float32, order='C')
        X.sort_indices()
        data = X.data.ctypes.data_as(POINTER(c_float))
        indices = X.indices.ctypes.data_as(POINTER(c_int32))
        indptr = X.indptr.ctypes.data_as(POINTER(c_int32))
        y = np.asarray(y, dtype=np.float32, order='C')
        label = y.ctypes.data_as(POINTER(c_float))
        in_groups, num_groups = self._construct_groups(groups)
        group_label = (c_float * len(set(y)))()
        n_class = (c_int * 1)()
        n_class[0] = self.num_class
        tree_per_iter_ptr = (c_int * 1)()
        self.model = (c_long * 1)()
        thundergbm.sparse_train_scikit(X.shape[0], data, indptr, indices, label, self.depth, self.n_trees,
                                       self.n_gpus, c_float(self.min_child_weight), c_float(self.lambda_tgbm),
                                       c_float(self.gamma),
                                       self.max_num_bin, self.verbose, c_float(self.column_sampling_rate), self.bagging,
                                       self.n_parallel_trees, c_float(self.learning_rate),
                                       self.objective.encode('utf-8'),
                                       n_class, self.tree_method.encode('utf-8'), byref(self.model), tree_per_iter_ptr,
                                       group_label,
                                       in_groups, num_groups)
        self.num_class = n_class[0]
        self.tree_per_iter = tree_per_iter_ptr[0]
        self.group_label = [group_label[idx] for idx in range(len(set(y)))]
        if self.model is None:
            print("The model returned is empty!")
            exit()

    def predict(self, X, groups=None):
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
        if(self.objective != 'multi:softprob'):
            self.predict_label_ptr = (c_float * X.shape[0])()
        else:
            temp_size = X.shape[0] * self.num_class
            self.predict_label_ptr = (c_float * temp_size)()
        if self.group_label is not None:
            group_label = (c_float * len(self.group_label))()
            group_label[:] = self.group_label
        else:
            group_label = None
        in_groups, num_groups = self._construct_groups(groups)
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
            group_label,
            in_groups, num_groups, self.verbose
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


    def cv(self, X, y, folds=None, nfold=5, shuffle=True, seed=0):
        if self._impl == 'ranker':
            print("Cross-validation for ## Ranker ## have not been supported yep..")
            return

        sparse = sp.isspmatrix(X)
        if sparse is False:
            X = sp.csr_matrix(X)
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')
        y = np.asarray(y, dtype=np.float32, order='C')
        n_instances = X.shape[0]
        if folds is not None:
            #use specified validation set
            train_idset = [x[0] for x in folds]
            test_idset = [x[1] for x in folds]
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(n_instances)
            else:
                randidx = np.arange(n_instances)
            kstep = int(n_instances / nfold)
            test_idset = [randidx[i: i + kstep] for i in range(0, n_instances, kstep)]
            train_idset = [np.concatenate([test_idset[i] for i in range(nfold) if k != i]) for k in range(nfold)]
        # to be optimized: get score in fit; early stopping; more metrics;
        train_score_list = []
        test_score_list = []
        for k in range(nfold):
            X_train = X[train_idset[k],:]
            X_test = X[test_idset[k],:]
            y_train = y[train_idset[k]]
            y_test = y[test_idset[k]]
            self.fit(X_train, y_train)
            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)
            if self._impl == 'classifier':
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test,y_test_pred)
                train_score_list.append(train_score)
                test_score_list.append(test_score)
            elif self._impl == 'regressor':
                train_score = mean_squared_error(y_train,y_train_pred)
                test_score = mean_squared_error(y_test, y_test_pred)
                train_score_list.append(train_score)
                test_score_list.append(test_score)
        self.eval_res = {}
        if self._impl == 'classifier':
            self.eval_res['train-accuracy-mean']= statistics.mean(train_score_list)
            self.eval_res['train-accuracy-std']= statistics.stdev(train_score_list)
            self.eval_res['test-accuracy-mean'] = statistics.mean(test_score_list)
            self.eval_res['test-accuracy-std'] = statistics.stdev(test_score_list)
            print("mean train accuracy:%.6f+%.6f" %(statistics.mean(train_score_list), statistics.stdev(train_score_list)))
            print("mean test accuracy:%.6f+%.6f" %(statistics.mean(test_score_list), statistics.stdev(test_score_list)))
        elif self._impl == 'regressor':
            self.eval_res['train-RMSE-mean']= statistics.mean(train_score_list)
            self.eval_res['train-RMSE-std']= statistics.stdev(train_score_list)
            self.eval_res['test-RMSE-mean'] = statistics.mean(test_score_list)
            self.eval_res['test-RMSE-std'] = statistics.stdev(test_score_list)
            print("mean train RMSE:%.6f+%.6f" %(statistics.mean(train_score_list), statistics.stdev(train_score_list)))
            print("mean test RMSE:%.6f+%.6f" %(statistics.mean(test_score_list), statistics.stdev(test_score_list)))
        return self.eval_res

    def get_shap_trees(self):
        if self.objective != 'binary:logistic' and self.tree_per_iter != 1:
            raise RuntimeError("not supported yet!")
        n_tree_per_iter = 1
        trees = []
        n_total_trees = self.n_trees * self.tree_per_iter
        n_nodes_list = (c_int * n_total_trees)()
        thundergbm.get_n_nodes(byref(self.model), n_nodes_list, self.n_trees, self.tree_per_iter)
        for k in range(self.n_trees):
            tree_id = k * self.tree_per_iter
            n_nodes = n_nodes_list[tree_id]
            # print("in round ",k)
            # print("n_nodes ",n_nodes)
            # children_left = np.empty(n_nodes,dtype=np.int32)
            # children_right = np.empty(n_nodes,dtype=np.int32)
            # children_default = np.empty(n_nodes,dtype=np.int32)
            # features = np.empty(n_nodes,dtype=np.int32)
            # thresholds = np.empty(n_nodes,dtype=np.float32)
            # values = np.empty(n_nodes,dtype=np.float32)
            # node_sample_weight = np.empty(n_nodes,dtype=np.float32)

            trees.append({})
            trees[-1]['children_left'] = np.empty(n_nodes,dtype=np.int32)
            trees[-1]['children_right'] = np.empty(n_nodes,dtype=np.int32)
            trees[-1]['children_default'] = np.empty(n_nodes,dtype=np.int32)
            trees[-1]['feature'] = np.empty(n_nodes,dtype=np.int32)
            trees[-1]['threshold'] = np.empty(n_nodes,dtype=np.float32)
            trees[-1]['value'] = np.empty(n_nodes,dtype=np.float32)
            trees[-1]['node_sample_weight'] = np.empty(n_nodes,dtype=np.float32)

            # children_left = (c_int * n_nodes)()
            # children_right = (c_int * n_nodes)()
            # children_default = (c_int * n_nodes)()
            # features = (c_int * n_nodes)()
            # thresholds = (c_float * n_nodes)()
            # values = (c_float * n_nodes)()
            # node_sample_weight = (c_float * n_nodes)()
            thundergbm.get_a_tree(byref(self.model), tree_id, n_nodes, trees[-1]['children_left'].ctypes.data_as(POINTER(c_int32)),
                                  trees[-1]['children_right'].ctypes.data_as(POINTER(c_int32)),
                                  trees[-1]['children_default'].ctypes.data_as(POINTER(c_int32)),
                                  trees[-1]['feature'].ctypes.data_as(POINTER(c_int32)),
                                  trees[-1]['threshold'].ctypes.data_as(POINTER(c_float)),
                                  trees[-1]['value'].ctypes.data_as(POINTER(c_float)),
                                  trees[-1]['node_sample_weight'].ctypes.data_as(POINTER(c_float)))
            # thundergbm.get_a_tree(byref(self.model), k, n_nodes, children_left, children_right, children_default,
            #                       features, thresholds, values, node_sample_weight)

            # print(children_left)
            # trees[-1]['children_left']=np.copy(children_left)
            # trees[-1]['children_right']=np.copy(children_right)
            # trees[-1]['children_default']=np.copy(children_default)
            # trees[-1]['feature']=np.copy(features)
            # trees[-1]['threshold']=np.copy(thresholds)
            # trees[-1]['value']=np.copy(values).reshape(-1,1)
            # trees[-1]['node_sample_weight']=np.copy(node_sample_weight)

            trees[-1]['value']=trees[-1]['value'].reshape(-1,1)

        # print(len(trees))
        import shap.explainers.tree as shap_tree
        shap_trees = []
        for k in range(self.n_trees * n_tree_per_iter):
            shap_trees.append(shap_tree.Tree(trees[k]))

        return shap_trees




class TGBMClassifier(TGBMModel, ThundergbmClassifierBase):
    _impl = 'classifier'
    def __init__(self, depth=6, n_trees=40,
                 n_gpus=1, min_child_weight=1.0, lambda_tgbm=1.0, gamma=1.0, max_num_bin=255,
                 verbose=1, column_sampling_rate=1.0, bagging=0,
                 n_parallel_trees=1, learning_rate=1.0, objective="multi:softmax",
                 num_class=2, tree_method="auto"):
        super().__init__(depth=depth, n_trees=n_trees,
                         n_gpus=n_gpus, min_child_weight=min_child_weight, lambda_tgbm=lambda_tgbm, gamma=gamma,
                         max_num_bin=max_num_bin,
                         verbose=verbose, column_sampling_rate=column_sampling_rate, bagging=bagging,
                         n_parallel_trees=n_parallel_trees, learning_rate=learning_rate, objective=objective,
                         num_class=num_class, tree_method=tree_method)


class TGBMRegressor(TGBMModel, ThundergbmRegressorBase):
    _impl = 'regressor'
    def __init__(self, depth=6, n_trees=40,
                 n_gpus=1, min_child_weight=1.0, lambda_tgbm=1.0, gamma=1.0, max_num_bin=255,
                 verbose=1, column_sampling_rate=1.0, bagging=0,
                 n_parallel_trees=1, learning_rate=1.0, objective="reg:linear",
                 num_class=1, tree_method="auto"):
        super().__init__(depth=depth, n_trees=n_trees,
                         n_gpus=n_gpus, min_child_weight=min_child_weight, lambda_tgbm=lambda_tgbm, gamma=gamma,
                         max_num_bin=max_num_bin,
                         verbose=verbose, column_sampling_rate=column_sampling_rate, bagging=bagging,
                         n_parallel_trees=n_parallel_trees, learning_rate=learning_rate, objective=objective,
                         num_class=num_class, tree_method=tree_method)


class TGBMRanker(TGBMModel, ThundergbmRegressorBase):
    _impl = 'ranker'
    def __init__(self, depth=6, n_trees=40,
                 n_gpus=1, min_child_weight=1.0, lambda_tgbm=1.0, gamma=1.0, max_num_bin=255,
                 verbose=1, column_sampling_rate=1.0, bagging=0,
                 n_parallel_trees=1, learning_rate=1.0, objective="reg:linear",
                 num_class=1, tree_method="auto"):
        super().__init__(depth=depth, n_trees=n_trees,
                         n_gpus=n_gpus, min_child_weight=min_child_weight, lambda_tgbm=lambda_tgbm, gamma=gamma,
                         max_num_bin=max_num_bin,
                         verbose=verbose, column_sampling_rate=column_sampling_rate, bagging=bagging,
                         n_parallel_trees=n_parallel_trees, learning_rate=learning_rate, objective=objective,
                         num_class=num_class, tree_method=tree_method)
