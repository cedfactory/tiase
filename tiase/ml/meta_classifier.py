from sklearn.ensemble import VotingClassifier
from . import analysis

def PrepareModelsForMetaClassifierVoting(models):
    formatted_estimators = []
    for model in models:
        name = model["name"]
        estimator = model["classifier"].model
        formatted_estimators.append((name, estimator))
    return formatted_estimators


class MetaClassifierVoting:
    def __init__(self, classifiers, X_train, y_train, X_test, y_test, params=None):
        '''
        format for classifiers : [('classifier1_name', classifier1), ('classifier2_name', classifier2)]
        '''
        self.classifiers = classifiers

        self.voting = "soft"
        if params:
            self.voting = params.get("voting", self.voting)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def build(self):
        print(self.classifiers)
        self.model = VotingClassifier(estimators=self.classifiers, voting=self.voting)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, x):
        self.model.predict(x)

    def get_analysis(self):
        self.y_test_pred = self.model.predict(self.X_test)
        self.y_test_prob = self.model.predict_proba(self.X_test)
        self.y_test_prob = self.y_test_prob[:, 1]
        self.analysis = analysis.classification_analysis(self.X_test, self.y_test, self.y_test_pred, self.y_test_prob)
        return self.analysis
