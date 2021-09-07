import pandas as pd
from lib.fimport import synthetic
from lib.findicators import findicators
from lib.ml import classifier_naive,classifier_lstm,classifier_svc,classifier_xgboost
import pytest

class TestMlClassifier:

    def get_dataframe(self):
        y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)
        df = findicators.add_technical_indicators(df, ["target"])
        df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])
        df.dropna(inplace = True)
        return df

    def test_classifier_alwayssameclass(self):
        df = self.get_dataframe()

        model = classifier_naive.ClassifierAlwaysSameClass(df, "target")
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(0.498007, 0.00001))
        assert(model_analysis["recall"] == pytest.approx(1., 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.664893, 0.00001))

    def test_classifier_alwaysasprevious(self):
        df = self.get_dataframe()

        model = classifier_naive.ClassifierAlwaysAsPrevious(df, "target")
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(0.968, 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.968, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.968, 0.00001))

    def test_classifier_lstm1(self):
        df = self.get_dataframe()
        
        model = classifier_lstm.ClassifierLSTM1(df, "target", params={'epochs': 20})
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(0.992647, 0.00001))
        assert(model_analysis["recall"] == pytest.approx(1., 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.996309, 0.00001))

    def test_classifier_lstm2(self):
        df = self.get_dataframe()
        
        model = classifier_lstm.ClassifierLSTM2(df, "target", params={'epochs': 20})
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.985185, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.992537, 0.00001))

    def test_classifier_lstm3(self):
        df = self.get_dataframe()
        
        model = classifier_lstm.ClassifierLSTM3(df, "target", params={'epochs': 20})
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.955555, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.977272, 0.00001))

    def test_classifier_lstm_hao2020(self):
        df = self.get_dataframe()
        
        model = classifier_lstm.ClassifierLSTMHao2020(df, "target", params={'epochs': 20})
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.985185, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.992537, 0.00001))

    def test_classifier_bilstm(self):
        df = self.get_dataframe()

        model = classifier_lstm.ClassifierBiLSTM(df, "target")
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.985185, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.992537, 0.00001))

    def _test_classifier_cnnbilstm(self):
        df = self.get_dataframe()

        model = classifier_lstm.ClassifierCNNBiLSTM(df, "target")
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(0.977443, 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.962962, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.970149, 0.00001))

    def test_classifier_svc(self):
        df = self.get_dataframe()

        model = classifier_svc.ClassifierSVC(df, "target")
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(0.992, 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.992, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.992, 0.00001))

    def test_classifier_xgboost(self):
        df = self.get_dataframe()

        model = classifier_xgboost.ClassifierXGBoost(df, "target")
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(1., 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(1., 0.00001))
