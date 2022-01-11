import xml.etree.cElementTree as ET
from . import classifier,analysis
from sklearn.model_selection import GridSearchCV
import json

class HPTGridSearch(classifier.Classifier):
    """
    reference : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """

    def __init__(self, params=None):

        self.classifier = None
        self.param_grid = None
        self.scoring = 'roc_auc' # \in {'roc_auc', ''}
        if params:
            self.classifier = params.get("classifier", self.classifier)
            self.scoring = params.get("scoring", self.scoring)
            self.param_grid = self.classifier.get_param_grid()
            self.param_grid = params.get("param_grid", self.param_grid)

    def get_name(self):
        return "grid search"

    def get_param_grid(self):
        return {}
        
    def build(self):
        self.model = GridSearchCV(estimator=self.classifier.get_model(), param_grid=self.param_grid, scoring=self.scoring)

    def fit(self, data_splitter):
        self.result = None
        self.data_splitter = data_splitter
        self.build()
        self.model.fit(self.data_splitter.X_train, self.data_splitter.y_train)
        print(self.model.best_params_)
        return self.model.best_params_

    def get_analysis(self):
        if self.result:
            return self.result
        y_test_pred, y_test_prob = classifier.get_pred_and_prob_with_predict_pred_and_predict_proba(self.model, self.data_splitter)
        self.result = analysis.classification_analysis(self.data_splitter.X_test, self.data_splitter.y_test, y_test_pred, y_test_prob)
        self.result["best_param"] = self.model.best_params_
        return self.result

    def save(self, filename):

        model_analysis = self.get_analysis()

        root = ET.Element("model")
        ET.SubElement(root, "param_grid").text = str(self.param_grid)
        ET.SubElement(root, "best_param").text = str(model_analysis["best_param"])
        ET.SubElement(root, "accuracy").text = "{:.2f}".format(model_analysis["accuracy"])
        ET.SubElement(root, "precision").text = "{:.2f}".format(model_analysis["precision"])
        ET.SubElement(root, "recall").text = "{:.2f}".format(model_analysis["recall"])
        ET.SubElement(root, "f1_score").text = "{:.2f}".format(model_analysis["f1_score"])
        tree = ET.ElementTree(root)

        xmlfilename = filename+'.xml'
        tree.write(xmlfilename)

    def load(self, filename):

        tree = ET.parse(filename+'.xml')
        root = tree.getroot()

        self.result = {}
        best_param = root.find('best_param').text
        best_param = best_param.replace("'", "\"")
        self.result["best_param"] = json.loads(best_param)
        self.result["accuracy"] = float(root.find('accuracy').text)
        self.result["precision"] = float(root.find('precision').text)
        self.result["recall"] = float(root.find('recall').text)
        self.result["f1_score"] = float(root.find('f1_score').text)
