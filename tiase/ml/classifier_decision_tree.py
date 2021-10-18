from sklearn.tree import DecisionTreeClassifier
from . import classifier,toolbox,analysis
from ..findicators import findicators

'''
DecisionTree

Input:
- seq_len = 21
'''
class ClassifierDecisionTree(classifier.Classifier):
    def __init__(self, dataframe, target, data_splitter, params = None):
        super().__init__(dataframe, target, data_splitter, params)

    def build(self):
        self.model = DecisionTreeClassifier()

    def fit(self):
        self.X_train = self.data_splitter.X_train
        self.y_train = self.data_splitter.y_train
        self.X_test = self.data_splitter.X_test
        self.y_test = self.data_splitter.y_test
        self.x_normaliser = self.data_splitter.normalizer
        
        self.build()
        self.model.fit(self.X_train,self.y_train)

    def get_analysis(self):
        self.y_test_pred = self.model.predict(self.X_test)
        self.y_test_prob = self.model.predict_proba(self.X_test)
        self.y_test_prob = self.y_test_prob[:, 1]
        self.analysis = analysis.classification_analysis(self.X_test, self.y_test, self.y_test_pred, self.y_test_prob)
        return self.analysis

    def save(self, filename):
        print("ClassifierDecisionTree.save() is not implemented")

    def load(self, filename):
        print("ClassifierDecisionTree.load() is not implemented")
