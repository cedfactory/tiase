import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *
import pytest

class TestMlLstmHao:

    def test_lstml_hao_basic(self):
        filename = "./lib/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)

        model = lstm_hao.LSTMHaoBasic(df)
        model.create_model(epochs = 10)

        analysis = model.get_analysis()

        assert(analysis["mape"] == pytest.approx(5.54, 0.01))
        assert(analysis["rmse"] == pytest.approx(81.60, 0.01))
        assert(analysis["mse"] == pytest.approx(6658.72, 0.01))

        prediction = model.predict()
        assert(prediction == pytest.approx(1423.31, 0.0001))
        

    def test_lstml_hao_trend(self):
        filename = "./lib/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)

        model = lstm_hao.LSTMHaoTrend(df)
        model.create_model(epochs = 10)

        analysis = model.get_analysis()

        #assert(analysis["mape"] == inf)
        assert(analysis["rmse"] == pytest.approx(22.41, 0.01))
        assert(analysis["mse"] == pytest.approx(502.38, 0.01))

        prediction = model.predict()
        assert(prediction == pytest.approx(1639.23, 0.0001))
