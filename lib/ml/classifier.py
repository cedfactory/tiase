import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from findicators import *
from ml import analysis
from ml import toolbox

from abc import ABCMeta, abstractmethod

class Classifier(object):
    __metaclass__=ABCMeta
    @abstractmethod
    def __build_model(self):
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
        self.analysis = analysis.classification_analysis(self.model, self.X_test, self.y_test)
        self.analysis["history"] = getattr(self, "history", None)
        return self.analysis

