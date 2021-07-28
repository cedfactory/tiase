from sklearn.svm import SVC
from ..ml import classifier,toolbox,analysis
from ..findicators import *

class ClassifierSVC(classifier.Classifier):
    def __init__(self, dataframe, name = ""):
        self.seq_len = 50
        self.df = dataframe

    def build_model(self):
        self.model = SVC(kernel="linear", C=0.025, probability=True)

    def create_model(self):
        self.set_train_test_data()
        self.build_model()
        self.model.fit(self.X_train,self.y_train)

    def get_analysis(self):
        self.y_test_pred = self.model.predict(self.X_test)
        self.y_test_prob = self.model.predict_proba(self.X_test)
        self.y_test_prob = self.y_test_prob[:, 1]
        self.analysis = analysis.classification_analysis(self.model, self.X_test, self.y_test, self.y_test_pred, self.y_test_prob)
        return self.analysis
