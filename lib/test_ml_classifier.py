import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *
import pytest

class TestMlClassifier:

    def get_dataframe(self):
        y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)
        df = findicators.add_technical_indicators(df, ["trend_1d"])
        df = findicators.remove_features(df, ["open","adj_close","low","high","volume"])
        df.dropna(inplace = True)
        return df


    def test_classifier_lstm1(self):
        df = self.get_dataframe()
        
        model = classifier_lstm.ClassifierLSTM1(df, params={'epochs': 20})
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(0.992647, 0.00001))
        assert(model_analysis["recall"] == pytest.approx(1., 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.996309, 0.00001))

    def test_classifier_lstm2(self):
        df = self.get_dataframe()
        
        model = classifier_lstm.ClassifierLSTM2(df, params={'epochs': 20})
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.985185, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.992537, 0.00001))

    def test_classifier_lstm3(self):
        df = self.get_dataframe()
        
        model = classifier_lstm.ClassifierLSTM3(df, params={'epochs': 20})
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.955555, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.977272, 0.00001))

    def test_classifier_lstm_hao2020(self):
        df = self.get_dataframe()
        
        model = classifier_lstm.ClassifierLSTM3(df, params={'epochs': 20})
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.955555, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.977272, 0.00001))

    def test_classifier_bilstm(self):
        df = self.get_dataframe()

        model = classifier_lstm.ClassifierBiLSTM(df)
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.992592, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.996282, 0.00001))

    def test_classifier_svc(self):
        df = self.get_dataframe()

        model = classifier_svc.ClassifierSVC(df)
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(0.991935, 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.991935, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.99193, 0.00001))

    def test_classifier_xgboost(self):
        df = self.get_dataframe()

        model = classifier_xgboost.ClassifierXGBoost(df)
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(1., 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(1., 0.00001))
