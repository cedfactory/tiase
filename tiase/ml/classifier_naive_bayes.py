from sklearn.naive_bayes import GaussianNB
from . import classifier,analysis

'''
ClassifierGaussianNB

Input:

ref : https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
'''
class ClassifierGaussianNB(classifier.Classifier):
    def __init__(self, dataframe, data_splitter, params = None):
        super().__init__(dataframe, data_splitter, params)

    def get_param_grid(self):
        return {}

    def get_model(self):
        return self.model

    def build(self):
        self.model = GaussianNB()

    def fit(self):
        self.build()
        self.model.fit(self.data_splitter.X_train,self.data_splitter.y_train)
        
    def get_analysis(self):
        y_test_pred, y_test_prob = classifier.get_pred_and_prob_with_predict_pred_and_predict_proba(self.model, self.data_splitter)
        return analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, y_test_pred, y_test_prob)

    def save(self, filename):
        print("ClassifierGaussianNB.save() is not implemented")

    def load(self, filename):
        print("ClassifierGaussianNB.load() is not implemented")
