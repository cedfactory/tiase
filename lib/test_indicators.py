from lib.fimport import *
from lib.findicators import *
import pandas as pd
import numpy as np

class TestIndicators:
    def test_number_colums(self):
        df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/AI.PA.csv")
        print(list(df.columns))
        assert(len(list(df.columns)) == 6)
        df = findicators.add_technical_indicators(df, ["macd", "rsi_30", "cci_30", "dx_30"])
        assert(len(list(df.columns)) == 10)
        df = findicators.remove_features(df, ["open", "high", "low"])
        assert(len(list(df.columns)) == 7)

    def test_get_trend_close(self):
        data = {'close':[20, 21, 23, 19, 18, 24]}
        df = pd.DataFrame(data)
        df = findicators.add_technical_indicators(df, ["trend"])
        trends = df.loc[:,'trend'].values
        equal = np.array_equal(trends, [0, 1, 1, 0, 0, 1])
        assert(equal)

