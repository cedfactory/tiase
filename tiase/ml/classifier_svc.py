from sklearn.svm import SVC
from . import classifier,analysis

'''
SVC

Input:
- C \in R, default=1.
- kernel \in {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
- degree \in N+, default=3
- gamma \in {'scale', 'auto'} or \in R, default='scale'

ref : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
'''
class ClassifierSVC(classifier.Classifier):
    def __init__(self, dataframe, target, data_splitter, params = None):
        super().__init__(dataframe, target, data_splitter, params)

        self.c = 1.
        self.kernel = "rbf"
        self.degree = 3
        self.gamma = 'scale'
        if params:
            self.c = params.get("c", self.c)
            self.kernel = params.get("kernel", self.kernel)
            self.degree = params.get("degree", self.degree)
            self.gamma = params.get("gamma", self.gamma)

    def get_param_grid(self):
        return {'C': [.8, .9, 1., 1.1, 1.2],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4, 5, 6],
        'gamma': ['scale', 'auto', .2, .4, .6, .8, 1., 1.2]
        }

    def get_model(self):
        return self.model

    def build(self):
        self.model = SVC(C=self.c, kernel=self.kernel, degree=self.degree, gamma=self.gamma, probability=True)

    def fit(self):
        self.build()
        self.model.fit(self.data_splitter.X_train,self.data_splitter.y_train)
        
    def get_analysis(self):
        y_test_pred, y_test_prob = classifier.get_pred_and_prob_with_predict_pred_and_predict_proba(self.model, self.data_splitter)
        return analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, y_test_pred, y_test_prob)

    def save(self, filename):
        print("ClassifierSVC.save() is not implemented")

    def load(self, filename):
        print("ClassifierSVC.load() is not implemented")
