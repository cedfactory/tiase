from . import analysis,toolbox

from abc import ABCMeta, abstractmethod

def set_train_test_cv_list(dataframe):
    list_df_train, list_df_test = toolbox.get_train_test_data_list_from_CV_WF_split_dataframe(dataframe, nb_split=20)
    return [list_df_train, list_df_test]

def set_train_test_data(dataframe, seq_len, split_index, target):
    # split the data
    x_train, y_train, x_test, y_test, x_normaliser = toolbox.get_train_test_data_from_dataframe(dataframe, seq_len, target, split_index, debug=True)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    return x_train, y_train, x_test, y_test, x_normaliser

class Classifier(metaclass = ABCMeta):
    
    def __init__(self, dataframe, target):
        self.df = dataframe
        self.seq_len = 21
        self.target = target

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def create_model(self):
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
