from xgboost import XGBClassifier
from . import classifier,analysis

'''
XGBoost

Input:
- seq_len = 21
- n_estimators \in N+, default=100
- max_depth \in N+, default=None
- learning_rate \in R, default=1
- objective \in {'binary:logistic'}, default='binary:logistic'
- booster \in {'gbtree', 'gblinear', 'dart'}, default=None
- tree_method {'exact', 'approx', 'hist', 'gpu_hist'}, \in , default=None

reference : https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
'''
class ClassifierXGBoost(classifier.Classifier):
    def __init__(self, params = None):
        super().__init__(params)

        self.n_estimators = 100
        self.max_depth = None
        self.learning_rate = 1
        self.objective = 'binary:logistic'
        self.booster = None
        self.tree_method = None
        if params:
            self.n_estimators = params.get("n_estimators", self.n_estimators)
            self.max_depth = params.get("max_depth", self.max_depth)
            self.learning_rate = params.get("learning_rate", self.learning_rate)
            self.objective = params.get("objective", self.objective)
            self.booster = params.get("booster", self.booster)
            self.tree_method = params.get("tree_method", self.tree_method)

    def get_model(self):
        return self.model

    def get_param_grid(self):
        return {'n_estimators': [50, 75, 100, 125, 150, 175, 200],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [.8, .9, 1., 1.1, 1.2],
        'objective': ['binary:logistic'],
        'booster': ['gbtree', 'gblinear', 'dart'],
        'tree_method': ['exact', 'approx', 'hist', 'gpu_hist'],
        }

    def build(self):
        self.model = XGBClassifier(use_label_encoder=False, n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, objective=self.objective, tree_method=self.tree_method, eval_metric='error')

    def fit(self, data_splitter):
        self.data_splitter = data_splitter
        self.build()
        self.model.fit(self.data_splitter.X_train,self.data_splitter.y_train)

    def get_analysis(self):
        y_test_pred, y_test_prob = classifier.get_pred_and_prob_with_predict_pred_and_predict_proba(self.model, self.data_splitter)
        return analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, y_test_pred, y_test_prob)

    def save(self, filename):
        print("ClassifierXGBoost.save() is not implemented")

    def load(self, filename):
        print("ClassifierXGBoost.load() is not implemented")
