import pandas as pd
import numpy as np
from tiase.fimport import synthetic
from tiase.findicators import findicators
from tiase.ml import data_splitter,classifiers_factory
import pytest

class TestMlClassifier:

    def get_dataframe(self):
        y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)
        df = findicators.add_technical_indicators(df, ["target"])
        df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])
        df.dropna(inplace = True)
        return df

    def get_data_splitter(self):
        df = self.get_dataframe()
        ds = data_splitter.DataSplitterTrainTestSimple(df, target="target", seq_len=21)
        ds.split(0.7)
        return ds

    def test_classifier_evaluate_cross_validation(self):
        df = self.get_dataframe()
        ds = self.get_data_splitter()

        model = classifiers_factory.ClassifiersFactory.get_classifier("lstm1", {'epochs': 5}, ds)
        ds = data_splitter.DataSplitterForCrossValidation(df.copy(), nb_splits=5)
        results = model.evaluate_cross_validation(ds, "target")
        print(results)

        equal = np.array_equal(results["accuracies"], [0.975, 0.975, 0.975, 0.975, 0.9625])
        assert(results["average_accuracy"] == pytest.approx(0.972499, 0.00001))
        assert(equal)

    def _test_classifier_common(self, model, expected_results, epsilon):
        model.fit()
        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(expected_results["precision"], epsilon))
        assert(model_analysis["recall"] == pytest.approx(expected_results["recall"], epsilon))
        assert(model_analysis["f1_score"] == pytest.approx(expected_results["f1_score"], epsilon))

    def test_classifier_alwayssameclass(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("same class", None, ds)
        self._test_classifier_common(model, {"precision":0.482142, "recall":1., "f1_score":0.650602}, 0.00001)

    def test_classifier_alwaysasprevious(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("as previous", None, ds)
        self._test_classifier_common(model, {"precision":0.962962, "recall":0.962962, "f1_score":0.962962}, 0.00001)

    def test_classifier_lstm1(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("lstm1", {'epochs': 20}, ds)
        self._test_classifier_common(model, {"precision":0.992647, "recall":1., "f1_score":0.996309}, 0.00001)

    def test_classifier_lstm2(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("lstm2", {'epochs': 20}, ds)
        self._test_classifier_common(model, {"precision":1., "recall":0.985185, "f1_score":0.992537}, 0.00001)

    def test_classifier_lstm3(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("lstm3", {'epochs': 20}, ds)
        self._test_classifier_common(model, {"precision":1., "recall":0.955555, "f1_score":0.977272}, 0.00001)

    def test_classifier_lstm_hao2020(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("lstmhao2020", {'epochs': 20}, ds)
        self._test_classifier_common(model, {"precision":1., "recall":0.985185, "f1_score":0.992537}, 0.1)

    def test_classifier_bilstm(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("bilstm", None, ds)
        self._test_classifier_common(model, {"precision":0.9851852, "recall":0.985185, "f1_score":0.985185}, 0.1)

    def test_classifier_cnnbilstm(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("cnnbilstm", None, ds)
        self._test_classifier_common(model, {"precision":0.984732, "recall":0.955555, "f1_score":0.969924}, 0.1)

    def test_classifier_svc(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("svc", {'kernel': 'linear', 'c': 0.025}, ds)
        self._test_classifier_common(model, {"precision":0.985185, "recall":0.985185, "f1_score":0.985185}, 0.00001)

    def test_classifier_xgboost(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("xgboost", {"n_estimators":100}, ds)
        self._test_classifier_common(model, {"precision":1., "recall":1., "f1_score":1.}, 0.00001)

    def test_classifier_decision_tree(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("decision tree", None, ds)
        self._test_classifier_common(model, {"precision":1., "recall":1., "f1_score":1.}, 0.00001)

    def test_classifier_mlp(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("mlp", {'hidden_layer_sizes': 80, 'random_state': 1}, ds)
        self._test_classifier_common(model, {"precision":1., "recall":0.992592, "f1_score":0.996282}, 0.00001)

    def test_classifier_gaussian_naive_bayes(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("gaussian naive bayes", None, ds)
        self._test_classifier_common(model, {"precision":0.948529, "recall":0.955555, "f1_score":0.952029}, 0.00001)

    def test_classifier_gaussian_process(self):
        ds = self.get_data_splitter()
        model = classifiers_factory.ClassifiersFactory.get_classifier("gaussian process", None, ds)
        self._test_classifier_common(model, {"precision":1., "recall":0.992592, "f1_score":0.996282}, 0.00001)
