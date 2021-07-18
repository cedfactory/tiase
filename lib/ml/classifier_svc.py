import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from sklearn.svm import SVC
from ml import classifier
from ml import toolbox
from ml import analysis
from findicators import *

class ClassifierSVC(classifier.Classifier):
    def __init__(self, dataframe, name = ""):
        self.seq_len = 50
        self.df = dataframe

        self.df = findicators.add_technical_indicators(self.df, ["trend_1d"])
        self.df = findicators.remove_features(self.df, ["open","low","high","adj_close","volume"])
        #print(self.df.head(15))
        self.df.dropna(inplace = True)

    def build_model(self):
        print("build_model analysis")
        self.model = SVC(kernel="linear", C=0.025)

    def create_model(self):
        print("create_model analysis")
        # split the data
        self.X_train, self.y_train, self.X_test, self.y_test, self.x_normaliser = toolbox.get_train_test_data_from_dataframe2(self.df, self.seq_len, 'trend_1d', .7)
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

        self.build_model()
        self.model.fit(self.X_train,self.y_train)

    def get_analysis(self):
        print("svc analysis")
        self.analysis = analysis.classification_analysis(self.model, self.X_test, self.y_test)
        return self.analysis


