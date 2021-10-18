from sklearn.svm import SVC
from . import classifier,toolbox,analysis
from ..findicators import findicators

'''
SVC

Input:
- seq_len = 21
- kernel = "linear"
- c = 0.025
'''       
class ClassifierSVC(classifier.Classifier):
    def __init__(self, dataframe, target, data_splitter, params = None):
        super().__init__(dataframe, target, data_splitter, params)

        self.kernel = "linear"
        self.c = 0.025
        if params:
            self.kernel = params.get("kernel", self.kernel)
            self.c = params.get("c", self.c)

    def build(self):
        self.model = SVC(kernel=self.kernel, C=self.c, probability=True)

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
        print("ClassifierSVC.save() is not implemented")

    def load(self, filename):
        print("ClassifierSVC.load() is not implemented")
