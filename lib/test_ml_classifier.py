import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *
import pytest

class TestMlClassifier:

    def test_classifier_lstm(self):
        y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)

        model = classifier_lstm.ClassifierLSTM(df, params={'epochs': 20})
        model.create_model()

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(0.97, 0.01))
        assert(model_analysis["recall"] == pytest.approx(0.96, 0.01))
        assert(model_analysis["f1_score"] == pytest.approx(0.96, 0.01))

    def test_classifier_svc(self):
        y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)

        modelsvc = classifier_svc.ClassifierSVC(df.copy())
        modelsvc.create_model()

        model_analysis = modelsvc.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(0.991935, 0.00001))
        assert(model_analysis["recall"] == pytest.approx(0.991935, 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(0.99193, 0.00001))

    def test_classifier_xgboost(self):
        y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)

        modelsvc = classifier_xgboost.ClassifierXGBoost(df.copy())
        modelsvc.create_model()

        model_analysis = modelsvc.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(1., 0.00001))
        assert(model_analysis["recall"] == pytest.approx(1., 0.00001))
        assert(model_analysis["f1_score"] == pytest.approx(1., 0.00001))
