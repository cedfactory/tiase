from . import analysis,toolbox,data_splitter

from abc import ABCMeta, abstractmethod

def set_train_test_cv_list(dataframe):
    ds = data_splitter.DataSplitterForCrossValidation(dataframe, nb_splits=5)
    list_df_train, list_df_test = ds.split()
    return [list_df_train, list_df_test]

def set_train_test_data(dataframe, seq_len, split_index, target):
    # split the data
    ds = data_splitter.DataSplitter(dataframe, target, seq_len)
    ds.split(split_index)
    y_train = ds.y_train.astype(int)
    y_test = ds.y_test.astype(int)
    return ds.X_train, y_train, ds.X_test, y_test, ds.normalizer

class Classifier(metaclass = ABCMeta):
    
    def __init__(self, dataframe, target, params = None):
        self.df = dataframe
        self.target = target

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
