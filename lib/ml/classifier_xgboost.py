import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from xgboost import XGBClassifier
from ml import classifier,toolbox,analysis
from findicators import *

class ClassifierXGBoost(classifier.Classifier):
    def __init__(self, dataframe, name = ""):
        self.seq_len = 50
        self.df = dataframe

    def __build_model(self):
        self.model = XGBClassifier()

    def create_model(self):
        self.set_train_test_data()
        self.__build_model()
        self.model.fit(self.X_train,self.y_train)
