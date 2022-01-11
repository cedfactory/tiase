import pandas as pd
import numpy as np
from tiase.fimport import synthetic,fimport
from tiase.findicators import findicators
from tiase.fdatapreprocessing import fdataprep
from tiase.ml import data_splitter,classifiers_factory
from tiase import alfred
import pytest

class TestMlMultiClassifier:

    def get_dataframe_multiclass(self):
        y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)
        df = findicators.add_technical_indicators(df, ["target"])
        df["target"] = pd.cut(df["close"],
                bins=[-1.1, -.8, .8, 1.1], 
                labels=[0., 1., 2.])
        df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])
        df.dropna(inplace = True)
        return df

    def get_data_splitter_multiclass(self):
        df = self.get_dataframe_multiclass()
        ds = data_splitter.DataSplitterTrainTestSimple(df, target="target", seq_len=21)
        ds.split(0.7)
        return ds

    def _test_multiclassifier_common(self, model, expected_results, epsilon):
        ds = self.get_data_splitter_multiclass()
        model.fit(ds)
        model_analysis = model.get_analysis()

        assert(model_analysis["accuracy"] == pytest.approx(expected_results["accuracy"], epsilon))
        assert(model_analysis["precision"] == pytest.approx(expected_results["precision"], epsilon))
        assert(model_analysis["recall"] == pytest.approx(expected_results["recall"], epsilon))
        assert(model_analysis["f1_score"] == pytest.approx(expected_results["f1_score"], epsilon))
        assert(np.array_equal(model_analysis["confusion_matrix"], expected_results["confusion_matrix"]))

    def test_multiclassifier_decision_tree(self):
        model = classifiers_factory.ClassifiersFactory.get_classifier("decision tree")
        expected_confusion_matrix = np.array([[ 65,   0,   0],
                                            [  0, 164,   0],
                                            [  0,   0,  51]])
        self._test_multiclassifier_common(model, {"accuracy":1., "precision":1., "recall":1., "f1_score":1., "confusion_matrix":expected_confusion_matrix}, 0.00001)

    def test_multiclassifier_lstm1(self):
        model = classifiers_factory.ClassifiersFactory.get_classifier("lstm1", {'epochs': 10})
        expected_confusion_matrix = np.array([[ 65,   0,   0],
                                            [  3, 161,   0],
                                            [  0,   3,  48]])
        self._test_multiclassifier_common(model, {"accuracy":0.978571, "precision":0.979196, "recall":0.974294, "f1_score":0.976282, "confusion_matrix":expected_confusion_matrix}, 0.00001)
