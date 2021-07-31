from xgboost import XGBClassifier
from ..ml import classifier,toolbox,analysis
from ..findicators import *

class ClassifierXGBoost(classifier.Classifier):
    def __init__(self, dataframe, name = ""):
        self.seq_len = 50
        self.df = dataframe

    def build_model(self):
        self.model = XGBClassifier(use_label_encoder=False)

    def create_model(self):
        self.set_train_test_data()
        self.build_model()
        self.model.fit(self.X_train,self.y_train, eval_metric="logloss")

    def get_analysis(self):
        self.y_test_pred = self.model.predict(self.X_test)
        self.y_test_prob = self.model.predict_proba(self.X_test)
        self.y_test_prob = self.y_test_prob[:, 1]
        self.analysis = analysis.classification_analysis(self.model, self.X_test, self.y_test, self.y_test_pred, self.y_test_prob)
        return self.analysis
