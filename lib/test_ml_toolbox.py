import pandas as pd
import numpy as np
from lib.fimport import *
from lib.ml import toolbox

class TestMlToolbox:
    def test_get_trend_close(self):
        data = {'Values':[20, 21, 23, 19, 18, 24]}
        df = pd.DataFrame(data)
        dfWithTrends = toolbox.AddTrend(df, "Values", "Trend")
        trends = dfWithTrends.loc[:,'Trend'].values
        equal = np.array_equal(trends, [0, 1, 1, 0, 0, 1])
        assert(equal)

    def test_mape(self):
        y_true = np.array([1, 2, -4])
        y_pred = np.array([-39, -58, 76])
        mape = toolbox.get_mape(y_true, y_pred)
        assert(mape == 3000)

    def test_rmse(self):
        y_true = np.array([28, 20, 30])
        y_pred = np.array([26, 30, 16])
        rmse = toolbox.get_rmse(y_true, y_pred)
        assert(rmse == 10)
 
  
  
         

