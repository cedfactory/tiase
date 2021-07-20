import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from findicators import *
from ml import analysis

from abc import ABCMeta, abstractmethod

class Classifier(object):
    __metaclass__=ABCMeta
    @abstractmethod
    def build_model():
        pass

    @abstractmethod
    def create_model():
        pass

    @abstractmethod
    def get_analysis(self):
        self.analysis = analysis.classification_analysis(self.model, self.X_test, self.y_test)
        self.analysis["history"] = getattr(self, "history", None)
        return self.analysis


