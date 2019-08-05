from sklearn.model_selection import train_test_split
import utils.data_utils as du

class Dataset(object):

    def __init__(self, name, task, metric, X=None, y=None, get_func=None):
        """
        Please notice that the training set and test set are the same here.
        """
        group = None
        self.name = name
        self.task = task
        self.metric = metric
        if task == 'Ranking':
            if get_func is not None:
                X, y, group = get_func()
        else:
            if get_func is not None:
                X, y = get_func()
        self.X_train = X
        self.X_test = X
        self.y_train = y
        self.y_test = y
        self.groups = group

    def split_dataset(self, test_size=0.1):
        """
        Split the dataset in a certain proportion
        :param test_size: the proportion of test set
        :return:
        """
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X_train, self.X_test, test_size=test_size)


if __name__ == "__main__":
    X, y = du.get_higgs()
    dataset = Dataset(name='higgs', task='Regression', metric='RMSE', get_func=du.get_higgs)
    print(dataset.X_train.shape)
    print(dataset.y_test.shape)

    dataset.split_dataset(test_size=0.5)
    print(dataset.X_train.shape)
    print(dataset.y_test.shape)
