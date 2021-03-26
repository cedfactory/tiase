import pandas as pd
import numpy as np
from lib.fimport import *
from lib.ml import toolbox

class TestGetTendency:
    def test_trend_close(self):
        data = {'Values':[20, 21, 23, 19, 18, 24]}
        df = pd.DataFrame(data)
        dfWithTrends = toolbox.AddTrend(df, "Values", "Trend")
        trends = dfWithTrends.loc[:,'Trend'].values
        equal = np.array_equal(trends, [0, 1, 1, 0, 0, 1])
        assert(equal)

        

