from sklearn.svm import SVC
from . import classifier,toolbox,analysis
from ..findicators import findicators

'''
SVC

Input:
- seq_len = 21
- kernel \in {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
- c \in R, default=1.

ref : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
'''       
class ClassifierSVC(classifier.Classifier):
    def __init__(self, dataframe, target, data_splitter, params = None):
        super().__init__(dataframe, target, data_splitter, params)

        self.kernel = "rbf"
        self.c = 1.
        if params:
            self.kernel = params.get("kernel", self.kernel)
            self.c = params.get("c", self.c)

    def build(self):
        self.model = SVC(kernel=self.kernel, C=self.c, probability=True)

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
        print("ClassifierSVC.save() is not implemented")

    def load(self, filename):
        print("ClassifierSVC.load() is not implemented")
