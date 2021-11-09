import pandas as pd
from tiase.fimport import fimport
from tiase.ml import lstm_hao
import numpy as np
import pytest

class TestMlLstmHao:

    def test_get_train_test_data_from_dataframe_hao(self):
        data = [[10,1000], [2,500], [3,600], [2,800], [2,500], [0,700], [1,800], [3,500], [5,900], [7,1000]]
        df = pd.DataFrame(data, columns = ['indicator', 'adj_close'])
        df.index.name = 'Date'

        x_train, y_train, x_test, y_test, x_normaliser, y_normaliser = lstm_hao.get_train_test_data_from_dataframe_hao(df, 2, 'adj_close', 0.6)

        x_train_expected = np.array([[[1., 1. ], [0.2 ,0. ]], [[0.2, 0. ], [0.3, 0.2]], [[0.3, 0.2], [0.2, 0.6]], [[0.2, 0.6], [0.2, 0.]]])
        np.testing.assert_allclose(x_train, x_train_expected, 0.00001)

        y_train_expected = np.array([[0.3], [0.2], [0.2], [0. ]])
        np.testing.assert_allclose(y_train, y_train_expected, 0.00001)


    def test_lstml_hao_basic(self):
        filename = "./tiase/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)

        model = lstm_hao.LSTMHaoBasic(df)
        model.create_model(epochs = 10)

        analysis = model.get_analysis()

        assert(analysis["mape"] == pytest.approx(5.54, 0.01))
        assert(analysis["rmse"] == pytest.approx(81.60, 0.01))
        assert(analysis["mse"] == pytest.approx(6658.72, 0.01))

        prediction = model.predict()
        assert(prediction == pytest.approx(1423.31, 0.0001))

        model.save_model("hao")

        model2 = lstm_hao.LSTMHaoBasic(df, name="hao")
        prediction2 = model2.predict()
        assert(prediction2 == pytest.approx(1423.31, 0.0001))

    def test_lstml_hao_trend(self):
        filename = "./tiase/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)

        model = lstm_hao.LSTMHaoTrend(df)
        model.create_model(epochs = 10)

        analysis = model.get_analysis()

        assert(analysis["rmse"] == pytest.approx(22.41, 0.01))
        assert(analysis["mse"] == pytest.approx(502.38, 0.01))

        prediction = model.predict()
        assert(prediction == pytest.approx(1639.23, 0.0001))
