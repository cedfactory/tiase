from sklearn.svm import SVC
from . import classifier,toolbox,analysis
from ..findicators import findicators

class ClassifierSVC(classifier.Classifier):
    def __init__(self, dataframe, target, name = ""):
        super().__init__(dataframe, target)
        self.seq_len = 50

    def build_model(self):
        self.model = SVC(kernel="linear", C=0.025, probability=True)

    def create_model(self):
        self.X_train, self.y_train, self.X_test, self.y_test, self.x_normaliser = classifier.set_train_test_data(self.df, self.seq_len, self.target)
        self.build_model()
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
