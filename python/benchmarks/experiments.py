import utils.data_utils as du
from model.catboost_model import CatboostModel
from model.lightgbm_model import LightGBMModel
from model.xgboost_model import XGboostModel
from model.thundergbm_model import ThunderGBMModel
from model.datasets import Dataset
import utils.file_utils as fu
import pandas as pd
import math
import sys

class Experiment:
    def __init__(self, data_func, name, task, metric):
        self.data = Dataset(data_func())
        self.name = name
        self.task = task
        self.metric = metric


exp_datasets = [
    # Dataset(name='SUSY', task='Regression', metric='RMSE', get_func=du.get_susy),
    # Dataset(name='covtype', task='Regression', metric='RMSE', get_func=du.get_covtype),
    # Dataset(name='real-sim', task='Regression', metric='RMSE', get_func=du.get_realsim),
    # Dataset(name='cifar', task='Regression', metric='RMSE', get_func=du.get_cifar),
    # Dataset(name='higgs', task='Regression', metric='RMSE', get_func=du.get_higgs),
    # Dataset(name='higgs', task='Regression', metric='RMSE', get_func=du.get_higgs),
    # Dataset(name='log1p', task='Regression', metric='RMSE', get_func=du.get_log1p)
    Dataset(name='cifar10', task='Multiclass classification', metric='Accuracy', get_func=du.get_cifar10),
    # Dataset(name='news20', task='Multiclass classification', metric='Accuracy', get_func=du.get_news20),
    # Dataset(name='yahoo', task='Ranking', metric='NDCG', get_func=du.get_yahoo)
]

def higgs():
    return Dataset(name='higgs', task='Regression', metric='RMSE', get_func=du.get_higgs)

def log1p():
    return Dataset(name='log1p', task='Regression', metric='RMSE', get_func=du.get_log1p)

def cifar():
    return Dataset(name='cifar10', task='Multiclass classification', metric='Accuracy', get_func=du.get_cifar10)

def news20():
    return Dataset(name='news20', task='Multiclass classification', metric='Accuracy', get_func=du.get_news20)


def r_model(data, currentModel, df):
    elapsed, metric = currentModel.run_model(data=data)
    name = None
    if isinstance(currentModel, ThunderGBMModel):
        name = 'ThunderGBM(depth=' + str(currentModel.max_depth) + ')-' + currentModel.tree_method
    elif isinstance(currentModel, XGboostModel):
        use_gpu = 'gpu' if currentModel.use_gpu else 'cpu'
        name = 'XGBoost(depth=' + str(currentModel.max_depth) + ')-' + use_gpu
    elif isinstance(currentModel, CatboostModel):
        use_gpu = 'gpu' if currentModel.use_gpu else 'cpu'
        name = 'CatBoost(depth=' + str(currentModel.max_depth) + ')-' + use_gpu
    elif isinstance(currentModel, LightGBMModel):
        use_gpu = 'gpu' if currentModel.use_gpu else 'cpu'
        name = 'LightGBM(num_rounds=' + str(currentModel.num_rounds) + ')-' + use_gpu
    fu.add_data(df, name, data, elapsed, metric)

    return elapsed, metric


def do_exps():
    # xgb_model_cpu = XGboostModel()
    # xgb_model_cpu.use_gpu = False
    # xgb_model_cpu.use_exact = True
    df = pd.DataFrame()

    # model = XGboostModel()
    model = LightGBMModel()
    # model = CatboostModel()
    # model.use_gpu = False

    result_file = open(model.model_name() + '.result', 'a')
    for exp in exp_datasets:
        used_times = []
        metrics = []
        for i in range(1, 6, 1):
            result_str = "## Model: " + model.model_name() + " ##\n## Dataset: " + exp.name + " ##\n"
            ut, mc = r_model(exp, model, df)
            fu.write_results(df, 'result.csv', 'csv')
            used_times.append(ut)
            metrics.append(mc)
            result_str += ", ".join([str(x) for x in used_times]) + "\n"
            result_str += ", ".join([str(x) for x in metrics]) + "\n\n\n"
            print(result_str)
        result_file.write(result_str)
    result_file.close()

    # # cbModel_gpu = CatboostModel()
    # # cbModel_cpu = CatboostModel()
    # # lgbModel_gpu = LightGBMModel()
    # # lgbModel_cpu = LightGBMModel()
    # # xgbModel_gpu = XGboostModel()
    # # xgbModel_cpu = XGboostModel()
    # # tgbModel_hist = ThunderGBMModel()
    # tgbModel_exact = ThunderGBMModel()
    # tgbModel_exact.tree_method = 'auto'
    # # tgbModel_hist.tree_method = 'hist'
    # # xgbModel_cpu.use_gpu = False
    # # xgbModel_gpu.use_exact = False
    # # cbModel_cpu.use_gpu = False
    # # lgbModel_cpu.use_gpu = False
    #
    # df = pd.DataFrame()
    # # for i in [10 * int(math.pow(2, x)) for x in range(2, 3)]:
    # #     tgbModel.num_rounds = i
    # #     xgbModel_gpu.num_rounds = i
    # #     xgbModel_cpu.num_rounds = i
    # #     cbModel_cpu.num_rounds = i
    # #     cbModel_gpu.num_rounds = i
    # #     lgbModel_gpu.num_rounds = i
    # #
    # #     for exp in exp_datasets:
    # #         # r_model(exp, cbModel_gpu, df)
    # #         # r_model(exp, cbModel_cpu, df)
    # #         # r_model(exp, lgbModel_gpu, df)
    # #         # r_model(exp, lgbModel_cpu, df)
    # #         # r_model(exp, xgbModel_gpu, df)
    # #         # r_model(exp, xgbModel_cpu, df)
    # #         r_model(exp, tgbModel, df)
    # #         fu.write_results(df, 'result3.csv', 'csv')
    #
    #
    # for depth in range(14, 17, 2):
    #     tgbModel_exact.max_depth = depth
    #     for exp in exp_datasets:
    #         # r_model(exp, cbModel_gpu, df)
    #         # r_model(exp, cbModel_cpu, df)
    #         # r_model(exp, lgbModel_gpu, df)
    #         # r_model(exp, lgbModel_cpu, df)
    #         # r_model(exp, xgbModel_gpu, df)
    #         # r_model(exp, xgbModel_cpu, df)
    #         # tgbModel_hist.max_depth = depth
    #         # r_model(exp, tgbModel_hist, df)
    #         r_model(exp, tgbModel_exact, df)
    #         print("----------------->>>>>depth: " + str(depth))
    #         fu.write_results(df, 'result3.csv', 'csv')


def load_dataset(dataset_name):
    if dataset_name == 'higgs':
        return higgs()
    elif dataset_name == 'log1p':
        return log1p()
    elif dataset_name == 'cifar':
        return cifar()
    elif dataset_name == 'news20':
        return news20()


def do_exps_with_command(model_name, dataset_name, use_gpu=True):
    model = None
    result_path = ''
    if model_name == 'xgboost':
        model = XGboostModel()
        result_path = 'xgb.txt'
    elif model_name == 'catboost':
        model = CatboostModel()
        result_path = 'cbt.txt'
    elif model_name == 'lightgbm':
        model = LightGBMModel()
        result_path = 'lgb.txt'
    elif model_name == 'thundergbm':
        model = ThunderGBMModel()
        result_path = 'tgb.txt'
    else:
        print('illegal model name...')

    model.use_gpu = use_gpu
    dataset = load_dataset(dataset_name)

    df = pd.DataFrame()
    result_file = open(result_path, 'a')
    used_times = []
    metrics = []
    for i in range(1, 6, 1):
        result_str = "## Model: " + model.model_name() + " ##\n## Dataset: " \
                     + dataset.name + " ##\n"
        ut, mc = r_model(dataset, model, df)
        fu.write_results(df, 'result.csv', 'csv')
        used_times.append(ut)
        metrics.append(mc)
        result_str += ", ".join([str(x) for x in used_times]) + "\n"
        result_str += ", ".join([str(x) for x in metrics]) + "\n\n\n"
        print(result_str)
    result_file.write(result_str)
    result_file.close()

if __name__ == "__main__":
    print(sys.argv)
    model_name = sys.argv[1]
    use_gpu = True
    if sys.argv[2] == 'cpu':
        use_gpu = False
    dataset_name = sys.argv[3]
    do_exps_with_command(model_name, dataset_name, use_gpu)
    # do_exps()


