from ..findicators import *
from . import analysis,toolbox

from abc import ABCMeta, abstractmethod

class Classifier(object):
    __metaclass__=ABCMeta
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def set_train_test_data(self):
        # split the data
        self.X_train, self.y_train, self.X_test, self.y_test, self.x_normaliser = toolbox.get_train_test_data_from_dataframe2(self.df, self.seq_len, 'trend_1d', .7)
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def get_analysis(self):
        pass
