from xgboost import XGBClassifier
from . import classifier,toolbox,analysis
from ..findicators import findicators

'''
XGBoost

Input:
- seq_len = 21
'''
class ClassifierXGBoost(classifier.Classifier):
    def __init__(self, dataframe, target, data_splitter, params = None):
        super().__init__(dataframe, target, data_splitter, params)

    def build(self):
        self.model = XGBClassifier(use_label_encoder=False, objective='binary:logistic', eval_metric='error')

    def fit(self):
        self.build()
        self.model.fit(self.data_splitter.X_train,self.data_splitter.y_train)

    def get_analysis(self):
        self.y_test_pred = self.model.predict(self.data_splitter.X_test)
        self.y_test_prob = self.model.predict_proba(self.data_splitter.X_test)
        self.y_test_prob = self.y_test_prob[:, 1]
        self.analysis = analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, self.y_test_pred, self.y_test_prob)
        return self.analysis

    def save(self, filename):
        print("ClassifierXGBoost.save() is not implemented")

    def load(self, filename):
        print("ClassifierXGBoost.load() is not implemented")
