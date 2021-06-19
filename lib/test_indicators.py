from lib.fimport import *
from lib.findicators import *
import pandas as pd
import numpy as np

class TestIndicators:
    def test_number_colums(self):
        df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/AI.PA.csv")
        print(list(df.columns))
        assert(len(list(df.columns)) == 6)
        df = findicators.add_technical_indicators(df, ["macd", "rsi_30", "cci_30", "dx_30", "williams_%r", "stoch_%k", "stoch_%d", "er"])
        assert(len(list(df.columns)) == 14)
        df = findicators.remove_features(df, ["open", "high", "low"])
        assert(len(list(df.columns)) == 11)

    def test_get_trend_close(self):
        data = {'close':[20, 21, 23, 19, 18, 24, 25, 26, 27]}
        df = pd.DataFrame(data)
        df = findicators.add_technical_indicators(df, ["trend_1d","trend_4d"])
        trend1 = df.loc[:,'trend_1d'].values
        equal = np.array_equal(trend1, [0, 1, 1, 0, 0, 1, 1, 1, 1])
        assert(equal)
        trend4 = df.loc[:,'trend_4d'].values
        equal = np.array_equal(trend4, [np.NaN, np.NaN, np.NaN, 0.5, 0.5, 0.5, 0.5, 0.75, 1.], equal_nan = True)
        assert(equal)

    def test_get_trend_ratio(self):
        data = {'close':[20, 21, 23, 19, 18, 24, 25, 26, 27, 28]}
        df = pd.DataFrame(data)
        trend_ratio = findicators.get_trend_ratio(df)
        assert(trend_ratio == 70)
