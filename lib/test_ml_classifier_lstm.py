import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *
import pytest

class TestMlClassifierLstm:

    def test_classifier(self):
        y = synthetic.get_sinusoid(length=1000, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)

        model = classifier_lstm.LSTMClassification(df)
        model.create_model(epochs = 20)

        model_analysis = model.get_analysis()

        assert(model_analysis["precision"] == pytest.approx(0.97, 0.01))
        assert(model_analysis["recall"] == pytest.approx(0.96, 0.01))
        assert(model_analysis["f1_score"] == pytest.approx(0.96, 0.01))
