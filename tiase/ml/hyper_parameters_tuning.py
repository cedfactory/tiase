from . import classifier,analysis
from sklearn.model_selection import GridSearchCV
from abc import ABCMeta, abstractmethod

class HyperParametersTuning(metaclass = ABCMeta):
    """

    """
    def __init__(self, model, data_splitter, params=None):
        self.model = model
        self.data_splitter = data_splitter

    @abstractmethod
    def fit(self):
        pass

#class HPTGridSearch(HyperParametersTuning):
class HPTGridSearch(classifier.Classifier):
    """
    reference : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """

    def __init__(self, model, data_splitter, params=None):
        super().__init__(model, data_splitter, params)

        self.param_grid = None
        self.scoring = 'roc_auc' # \in {'roc_auc', ''}
        if params:
            self.param_grid = params.get("param_grid", self.param_grid)
            self.scoring = params.get("scoring", self.scoring)

        self.data_splitter = data_splitter
        self.model = model
        self.param_grid = self.model.get_param_grid()

    def build(self):
        self.tuner = GridSearchCV(estimator=self.model.get_model(), param_grid=self.param_grid, scoring=self.scoring)

    def fit(self):
        self.build()
        self.tuner.fit(self.data_splitter.X_train, self.data_splitter.y_train)
        return self.tuner.best_params_

    def get_analysis(self):
        self.y_test_pred = self.tuner.predict(self.data_splitter.X_test)
        self.y_test_prob = self.tuner.predict_proba(self.data_splitter.X_test)
        self.y_test_prob = self.y_test_prob[:, 1]
        self.analysis = analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, self.y_test_pred, self.y_test_prob)
        return self.analysis

    def save(self, filename):
        print("HPTGridSearch.save() is not implemented")

    def load(self, filename):
        print("HPTGridSearch.load() is not implemented")
