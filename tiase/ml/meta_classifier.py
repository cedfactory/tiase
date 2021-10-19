from sklearn.ensemble import VotingClassifier
from . import analysis

def prepare_models_for_meta_classifier_voting(models):
    formatted_estimators = []
    for model in models:
        name = model["name"]
        estimator = model["classifier"].model
        formatted_estimators.append((name, estimator))
    return formatted_estimators

'''
params :
- voting \in {‘hard’, ‘soft’}, default=’hard’

reference : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
'''
class MetaClassifierVoting:
    def __init__(self, classifiers, data_splitter, params=None):
        '''
        format for classifiers : [('classifier1_name', classifier1), ('classifier2_name', classifier2)]
        '''
        self.classifiers = classifiers

        self.voting = "hard"
        if params:
            self.voting = params.get("voting", self.voting)

        self.data_splitter = data_splitter


    def build(self):
        print(self.classifiers)
        self.model = VotingClassifier(estimators=self.classifiers, voting=self.voting)

    def fit(self):
        self.model.fit(self.data_splitter.X_train, self.data_splitter.y_train)

    def predict(self, x):
        self.model.predict(x)

    def get_analysis(self):
        self.y_test_pred = self.model.predict(self.data_splitter.X_test)
        self.y_test_prob = self.model.predict_proba(self.data_splitter.X_test)
        self.y_test_prob = self.y_test_prob[:, 1]
        self.analysis = analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, self.y_test_pred, self.y_test_prob)
        return self.analysis
