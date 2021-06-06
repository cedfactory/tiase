import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *
import pytest

class TestMlLstm:

    def test_lstml_basic(self):
        filename = "./lib/data/test/GOOG.csv"
        df = fimport.GetDataFrameFromCsv(filename)

        model = lstm_basic.LSTMBasic(df)
        model.create_model(epochs = 10)

        analysis = model.get_analysis()

        assert(analysis["mape"] == pytest.approx(5.47, 0.01))
        assert(analysis["rmse"] == pytest.approx(80.48, 0.01))
        assert(analysis["mse"] == pytest.approx(6477.77, 0.01))

        prediction = model.predict()
        assert(prediction == pytest.approx(1424.42, 0.0001))
        
