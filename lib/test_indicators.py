from lib.fimport import *
from lib.findicators import *
import pandas as pd
#from pandas._testing import assert_frame_equal
import numpy as np
import pytest
import datetime

class TestIndicators:
    def test_get_all_default_technical_indicators(self):
        ti = findicators.get_all_default_technical_indicators()
        assert(len(ti) == 23)
        expected_ti = ["trend_1d","macd","macds","macdh","bbands","rsi_30","cci_30","dx_30","williams_%r","stoch_%k","stoch_%d","er","stc","atr","adx","roc"]
        expected_ti.extend(["sma_5","sma_10","sma_15","sma_20"])
        expected_ti.extend(["ema_10","ema_20","ema_50"])
        assert(ti == expected_ti)

    def test_number_colums(self):
        df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/AI.PA.csv")
        print(list(df.columns))
        assert(len(list(df.columns)) == 6)

        # first set of technical indicators
        technical_indicators = ["macd", "rsi_30", "cci_30", "dx_30", "williams_%r", "stoch_%k", "stoch_%d", "er"]
        df = findicators.add_technical_indicators(df, technical_indicators)
        assert(len(list(df.columns)) == 14)

        df = findicators.remove_features(df, technical_indicators)
        assert(len(list(df.columns)) == 6)

        # second set of technical indicators
        technical_indicators = ["trend_1d", "sma_12", "ema_9", "bbands", "stc", "atr", "adx", "roc"]
        df = findicators.add_technical_indicators(df, technical_indicators)
        assert(len(list(df.columns)) == 16)

        technical_indicators.remove("bbands")
        technical_indicators.extend(["bb_upper", "bb_middle", "bb_lower"])
        df = findicators.remove_features(df, technical_indicators)
        assert(len(list(df.columns)) == 6)

        df = findicators.remove_features(df, ["open", "high"])
        assert(len(list(df.columns)) == 4)

        df = findicators.remove_features(df, ["fakecolumn", "low"])
        assert(len(list(df.columns)) == 3)

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
        trend_ratio, true_positive, true_negative, false_positive, false_negative = findicators.get_trend_info(df)
        assert(trend_ratio == pytest.approx(66.666, 0.001))
        assert(true_positive == pytest.approx(55.555, 0.001))
        assert(true_negative == pytest.approx(11.111, 0.001))
        assert(false_positive == pytest.approx(11.111, 0.001))
        assert(false_negative == pytest.approx(22.222, 0.001))

    def test_temporal_indicators(self):
        idx = pd.Index(pd.date_range("19991231", periods=10), name='Date')
        df = pd.DataFrame([1]*10, columns=["Foobar"], index=idx)
        df = findicators.add_temporal_indicators(df, "Date")

        expected_df = fimport.GetDataFrameFromCsv("./lib/data/test/temporal_indicators_reference.csv")
        #assert_frame_equal(df, expected_df, check_dtype=False) // don't work :(
        assert(df.equals(expected_df))
