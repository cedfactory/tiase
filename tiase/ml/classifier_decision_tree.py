from sklearn.tree import DecisionTreeClassifier
from . import classifier,toolbox,analysis
from ..findicators import findicators

'''
DecisionTree

Input:
- seq_len = 21
'''
class ClassifierDecisionTree(classifier.Classifier):
    def __init__(self, dataframe, target, params = None):
        super().__init__(dataframe, target, params)

    def build(self):
        self.model = DecisionTreeClassifier()

    def fit(self):
        self.X_train, self.y_train, self.X_test, self.y_test, self.x_normaliser = classifier.set_train_test_data(self.df, self.seq_len, 0.7, self.target)
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
