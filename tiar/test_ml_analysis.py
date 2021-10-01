import numpy as np
from tiar.ml import analysis

class TestMlAnalysis:

    def test_mape(self):
        y_true = np.array([1, 2, -4])
        y_pred = np.array([-39, -58, 76])
        mape = analysis.get_mape(y_true, y_pred)
        assert(mape == 3000)

    def test_rmse(self):
        y_true = np.array([28, 20, 30])
        y_pred = np.array([26, 30, 16])
        rmse = analysis.get_rmse(y_true, y_pred)
        assert(rmse == 10)
 
