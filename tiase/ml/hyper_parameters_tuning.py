from . import classifier,analysis
from sklearn.model_selection import GridSearchCV

class HPTGridSearch(classifier.Classifier):
    """
    reference : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """

    def __init__(self, params=None):

        self.classifier = None
        self.param_grid = None
        self.scoring = 'roc_auc' # \in {'roc_auc', ''}
        if params:
            self.param_grid = params.get("param_grid", self.param_grid)
            self.scoring = params.get("scoring", self.scoring)
            self.classifier = params.get("classifier", self.classifier)
            self.param_grid = self.classifier.get_param_grid()
        
    def get_name(self):
        return "grid search"

    def get_param_grid(self):
        return {}
        
    def build(self):
        self.model = GridSearchCV(estimator=self.classifier.get_model(), param_grid=self.param_grid, scoring=self.scoring)

    def fit(self, data_splitter):
        self.data_splitter = data_splitter
        self.build()
        self.model.fit(self.data_splitter.X_train, self.data_splitter.y_train)
        print(self.model.best_params_)
        return self.model.best_params_

    def get_analysis(self):
        y_test_pred, y_test_prob = classifier.get_pred_and_prob_with_predict_pred_and_predict_proba(self.model, self.data_splitter)
        return analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, y_test_pred, y_test_prob)

    def save(self, filename):
        print("HPTGridSearch.save() is not implemented")

    def load(self, filename):
        print("HPTGridSearch.load() is not implemented")
