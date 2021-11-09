from sklearn.tree import DecisionTreeClassifier
from . import classifier,analysis


'''
DecisionTree

Input:
- seq_len = 21
- criterion \in {'gini', 'entropy'}, default='gini'
- splitter \in {'best', 'random'}, default='best'
- max_depth = None
- random_state = None

reference : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
'''
class ClassifierDecisionTree(classifier.Classifier):
    def __init__(self, params = None):
        super().__init__(params)

        self.criterion = 'gini'
        self.splitter = 'best'
        self.max_depth = 10
        self.random_state = None
        if params:
            self.criterion = params.get("criterion", self.criterion)
            self.splitter = params.get("splitter", self.splitter)
            self.max_depth = params.get("max_depth", self.max_depth)
            self.random_state = params.get("random_state", self.random_state)

        if isinstance(self.max_depth, str):
            self.max_depth = int(self.max_depth)
        if isinstance(self.random_state, str):
            self.random_state = int(self.random_state)

    def get_param_grid(self):
        return {'max_depth': [2, 3, 4, 5, 6], 'criterion': ['gini', 'entropy'], 'splitter': ['best'], 'random_state': [1]}

    def get_model(self):
        return self.model

    def build(self):
        self.model = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter, max_depth=self.max_depth, random_state=self.random_state)

    def fit(self, data_splitter):
        self.data_splitter = data_splitter
        self.build()
        self.model.fit(self.data_splitter.X_train,self.data_splitter.y_train)

    def get_analysis(self):
        y_test_pred, y_test_prob = classifier.get_pred_and_prob_with_predict_pred_and_predict_proba(self.model, self.data_splitter)
        return analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, y_test_pred, y_test_prob)

    def save(self, filename):
        print("ClassifierDecisionTree.save() is not implemented")

    def load(self, filename):
        print("ClassifierDecisionTree.load() is not implemented")
