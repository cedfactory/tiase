from . import analysis,classifier
import numpy as np

from rich import print,inspect

class ClassifierAlwaysSameClass(classifier.Classifier):
    def __init__(self, data_splitter, params = None):
        super().__init__(data_splitter, params)
        self.class_to_return = 1
        self.seq_len = 50
        if params:
            self.seq_len = params.get("seq_len", self.seq_len)
            self.class_to_return = params.get("class_to_return", 1)

    def build(self):
        # nothing to build
        pass

    def fit(self):
        self.X_train = self.data_splitter.X_train
        self.y_train = self.data_splitter.y_train
        self.X_test = self.data_splitter.X_test
        self.y_test = self.data_splitter.y_test
        self.x_normaliser = self.data_splitter.normalizer
        
    def get_analysis(self):
        self.y_test_pred = np.empty(len(self.y_test))
        self.y_test_pred.fill(1)
        self.y_test_pred = self.y_test_pred.reshape(len(self.y_test_pred), 1)
        self.y_test_pred = self.y_test_pred.astype(int)
        self.y_test_prob = np.empty(len(self.y_test))
        self.y_test_prob.fill(1)

        self.analysis = analysis.classification_analysis(self.X_test, self.y_test, self.y_test_pred, self.y_test_prob)
        return self.analysis

    def save(self, filename):
        # nothing to save
        pass

    def load(self, filename):
        # nothing to load
        pass


class ClassifierAlwaysAsPrevious(classifier.Classifier):
    def __init__(self, data_splitter, params = None):
        super().__init__(data_splitter, params)
        self.seq_len = 50
        if params:
            self.seq_len = params.get("seq_len", self.seq_len)

    def build(self):
        # nothing to build
        pass

    def fit(self):
        self.X_train = self.data_splitter.X_train
        self.y_train = self.data_splitter.y_train
        self.X_test = self.data_splitter.X_test
        self.y_test = self.data_splitter.y_test
        self.x_normaliser = self.data_splitter.normalizer
        
    def get_analysis(self):
        self.y_test_pred = np.roll(self.y_test, 1)
        self.y_test_prob = np.empty(len(self.y_test))
        self.y_test_prob.fill(1.)

        self.analysis = analysis.classification_analysis(self.X_test, self.y_test, self.y_test_pred, self.y_test_prob)
        return self.analysis

    def save(self, filename):
        # nothing to save
        pass

    def load(self, filename):
        # nothing to load
        pass
