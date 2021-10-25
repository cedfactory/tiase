from . import analysis,toolbox,data_splitter

from abc import ABCMeta, abstractmethod


def get_pred_and_prob_with_predict_pred_and_predict_proba(model, data_splitter):
    y_test_pred = model.predict(data_splitter.X_test)
    y_test_prob = model.predict_proba(data_splitter.X_test)
    y_test_prob = y_test_prob[:, 1]
    return y_test_pred, y_test_prob

def set_train_test_data(dataframe, seq_len, split_index, target):
    # split the data
    ds = data_splitter.DataSplitterTrainTestSimple(dataframe, target, seq_len)
    ds.split(split_index)
    y_train = ds.y_train.astype(int)
    y_test = ds.y_test.astype(int)
    return ds.X_train, y_train, ds.X_test, y_test, ds.normalizer

class Classifier(metaclass = ABCMeta):
    
    def __init__(self, dataframe, data_splitter, params = None):
        self.df = dataframe
        self.data_splitter = data_splitter

        self.seq_len = 21
        if params:
            self.seq_len = params.get("seq_len", self.seq_len)


    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def get_analysis(self):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass
