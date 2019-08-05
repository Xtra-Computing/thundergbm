import os
import sys
import pandas as pd
from sklearn import datasets
import pickle
import numpy as np

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve  # pylint: disable=import-error,no-name-in-module
else:
    from urllib import urlretrieve  # pylint: disable=import-error,no-name-in-module

# to make a base directory of datasets
BASE_DATASET_DIR = 'datasets'
if not os.path.exists(BASE_DATASET_DIR):
    print("Made dir: %s" % BASE_DATASET_DIR)
    os.mkdir(BASE_DATASET_DIR)


def convert_to_plk(filename, plk_filename, data_url):
    if not os.path.isfile(plk_filename):
        print('Downloading dataset, please wait...')
        urlretrieve(data_url, filename)
        X, y = datasets.load_svmlight_file(filename)
        pickle.dump((X, y), open(plk_filename, 'wb'))
        os.remove(filename)
    print("----------- loading dataset %s -----------" % filename)
    X, y = pickle.load(open(plk_filename, 'rb'))
    print("----------- Finished loading dataset %s -----------" % filename)
    return X, y


# -------------------------------------------------------------------------------------------------
get_higgs_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2'  # pylint: disable=line-too-long
def get_higgs(num_rows=None):
    filename = os.path.join(BASE_DATASET_DIR, 'HIGGS.bz2')
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_higgs_url)

    return X.toarray(), y


# -------------------------------------------------------------------------------------------------
def get_covtype(num_rows=None):
    data = datasets.fetch_covtype()
    X = data.data
    y = data.target
    if num_rows is not None:
        X = X[0:num_rows]
        y = y[0:num_rows]

    return X, y


# -------------------------------------------------------------------------------------------------
get_e2006_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.train.bz2'  # pylint: disable=line-too-long
def get_e2006(num_rows=None):
    filename = os.path.join(BASE_DATASET_DIR, 'E2006.train')
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_e2006_url)

    return X.toarray(), y


# -------------------------------------------------------------------------------------------------
get_lop1p_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/log1p.E2006.train.bz2'
def get_log1p(num_rows=None):
    filename = os.path.join(BASE_DATASET_DIR, 'log1p.train.bz2')
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_lop1p_url)

    return X, y


# -------------------------------------------------------------------------------------------------
get_news20_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2'
def get_news20(num_rows=None):
    filename = os.path.join(BASE_DATASET_DIR, 'news20.binary.bz2')
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_news20_url)

    return X.toarray(), y


# -------------------------------------------------------------------------------------------------
get_realsim_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2'
def get_realsim(num_rows=None):
    filename = os.path.join(BASE_DATASET_DIR, 'real-sim.bz2')
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_realsim_url)

    return X.toarray(), y


# -------------------------------------------------------------------------------------------------
get_susy_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.bz2'
def get_susy(num_rows=None):
    filename = os.path.join(BASE_DATASET_DIR, 'susy.bz2')
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_susy_url)

    return X.toarray(), y


# -------------------------------------------------------------------------------------------------
get_epslion_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2'
def get_epsilon(num_rows=None):
    filename = os.path.join(BASE_DATASET_DIR, 'epsilon_normalized')
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_epslion_url)

    return X.toarray(), y


# -------------------------------------------------------------------------------------------------
get_cifar_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/cifar10.bz2'
def get_cifar(num_rows=None):
    filename = os.path.join(BASE_DATASET_DIR, 'cifar')
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_cifar_url)

    return X.toarray(), y


# -------------------------------------------------------------------------------------------------
get_ins_url = ''
def get_ins(num_rows=None):
    filename = BASE_DATASET_DIR + '/' + 'ins'
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_ins_url)

    return X.toarray(), y


get_cifar_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/cifar10.bz2'
def get_cifar10(num_rows=None):
    filename = BASE_DATASET_DIR + '/' + 'cifar10'
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_cifar_url)

    return X.toarray(), y



get_news20_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/news20.bz2'
def get_news20(num_rows=None):
    filename = BASE_DATASET_DIR + '/' + 'news20.bz2'
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_news20_url)

    return X.toarray(), y


def load_groups(filename):
    groups = []
    with open(filename) as f:
        for line in f:
            if line.strip() != '':
                groups.append(int(line.strip()))

    return np.asarray(groups)


get_yahoo_url = ''
def get_yahoo():
    filename = BASE_DATASET_DIR + "/" + 'yahoo-ltr-libsvm'
    group_filename = BASE_DATASET_DIR + "/" + 'yahoo-ltr-libsvm.group'
    plk_filename = filename + '.plk'
    X, y = convert_to_plk(filename, plk_filename, data_url=get_yahoo_url)
    groups = load_groups(group_filename)

    return X.toarray(), y, groups

if __name__ == "__main__":
    X, y, groups = get_yahoo();
    print(X.shape)
    print(y.shape)