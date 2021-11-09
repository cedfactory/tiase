from sklearn.ensemble import VotingClassifier
from . import analysis
from . import classifier

'''
params :
- voting \in {‘hard’, ‘soft’}, default=’hard’

reference : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
'''
class MetaClassifierVoting(classifier.Classifier):
    def __init__(self, params=None):
        '''
        format for params["classifiers"] : [('classifier1_name', classifier1), ('classifier2_name', classifier2)]
        '''
        self.classifiers = None
        self.voting = "soft"
        if params:
            self.classifiers = params.get("classifiers", self.classifiers)
            self.voting = params.get("voting", self.voting)

        # hack : classifiers received as parameters are Classifier objects.
        # since VotingClassifier deals with scikit models, we extract models from Classifier objects.
        self.classifiers = [(classifier[0], classifier[1].model) for classifier in self.classifiers]

    def get_param_grid(self):
        return {}
        
    def build(self):
        self.model = VotingClassifier(estimators=self.classifiers, voting=self.voting)

    def fit(self, data_splitter):
        self.data_splitter = data_splitter
        self.build()
        self.model.fit(self.data_splitter.X_train, self.data_splitter.y_train)

    def predict(self, x):
        self.model.predict(x)

    def get_analysis(self):
        y_test_pred, y_test_prob = classifier.get_pred_and_prob_with_predict_pred_and_predict_proba(self.model, self.data_splitter)
        return analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, y_test_pred, y_test_prob)

    def save(self, filename):
        print("MetaClassifierVoting.save() is not implemented")

    def load(self, filename):
        print("MetaClassifierVoting.load() is not implemented")
