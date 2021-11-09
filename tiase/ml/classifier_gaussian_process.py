from sklearn.gaussian_process import GaussianProcessClassifier
from . import classifier,analysis

'''
ClassifierGaussianProcess

Input:
- kernel \in , default=None
- optimizer \in {'fmin_l_bfgs_b', callable}, default='fmin_l_bfgs_b'

ref : https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
'''
class ClassifierGaussianProcess(classifier.Classifier):
    def __init__(self, params = None):
        super().__init__(params)

        self.kernel = None
        self.optimizer = 'fmin_l_bfgs_b'
        if params:
            self.kernel = params.get("kernel", self.kernel)
            self.optimizer = params.get("optimizer", self.optimizer)

    def get_param_grid(self):
        return {'kernel': [None],
        'optimizer': ['fmin_l_bfgs_b']
        }

    def get_model(self):
        return self.model

    def build(self):
        self.model = GaussianProcessClassifier(kernel=self.kernel, optimizer=self.optimizer)

    def fit(self, data_splitter):
        self.data_splitter = data_splitter
        self.build()
        self.model.fit(self.data_splitter.X_train,self.data_splitter.y_train)
        
    def get_analysis(self):
        y_test_pred, y_test_prob = classifier.get_pred_and_prob_with_predict_pred_and_predict_proba(self.model, self.data_splitter)
        return analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, y_test_pred, y_test_prob)

    def save(self, filename):
        print("ClassifierGaussianProcess.save() is not implemented")

    def load(self, filename):
        print("ClassifierGaussianProcess.load() is not implemented")
