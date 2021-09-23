from lib.fimport import fimport
from lib.findicators import findicators
from lib.fdatapreprocessing import fdataprep
import pandas as pd
# todo : try to use from pandas._testing import assert_frame_equal
import numpy as np
import pytest
import datetime

class TestIndicators:
    
    def get_real_dataframe(self):
        filename = "./lib/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)
        df = findicators.normalize_column_headings(df)
        return df

    def test_get_all_default_technical_indicators(self):
        ti = findicators.get_all_default_technical_indicators()
        assert(len(ti) == 28)
        expected_ti = ["trend_1d","macd","macds","macdh","bbands","rsi_30","cci_30","dx_30","williams_%r","stoch_%k","stoch_%d","er","stc","atr","adx","roc","mom","simple_rtn"]
        expected_ti.extend(["wma_5","wma_10","wma_15"])
        expected_ti.extend(["sma_5","sma_10","sma_15","sma_20"])
        expected_ti.extend(["ema_10","ema_20","ema_50"])
        assert(ti == expected_ti)

    def test_number_colums(self):
        df = fimport.get_dataframe_from_csv("./lib/data/CAC40/AI.PA.csv")
        assert(len(list(df.columns)) == 6)

        # first set of technical indicators
        technical_indicators = ["macd", "macds", "macdh", "rsi_30", "cci_30", "dx_30", "williams_%r", "stoch_%k", "stoch_%d", "er"]
        df = findicators.add_technical_indicators(df, technical_indicators)
        assert(len(list(df.columns)) == 16)

        df = findicators.remove_features(df, technical_indicators)
        assert(len(list(df.columns)) == 6)

        # second set of technical indicators
        technical_indicators = ["trend_1d", "sma_12", "ema_9", "bbands", "stc", "atr", "adx", "roc", "wma_5", "mom", "simple_rtn"]
        df = findicators.add_technical_indicators(df, technical_indicators)
        assert(len(list(df.columns)) == 19)

        technical_indicators.remove("bbands")
        technical_indicators.extend(["bb_upper", "bb_middle", "bb_lower"])
        df = findicators.remove_features(df, technical_indicators)
        assert(len(list(df.columns)) == 6)

        df = findicators.remove_features(df, ["open", "high"])
        assert(len(list(df.columns)) == 4)

        df = findicators.remove_features(df, ["fakecolumn", "low"])
        assert(len(list(df.columns)) == 3)

    def test_get_trend_close(self):
        data = {'close':[20, 21, 23, 19, 18, 24, 25, 26, 16, -10, -15, -18, -15, -8]}
        df = pd.DataFrame(data)
        df = findicators.add_technical_indicators(df, ["trend_1d","trend_4d"])
        trend1 = df.loc[:,'trend_1d'].values
        equal = np.array_equal(trend1, [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        assert(equal)
        trend4 = df.loc[:,'trend_4d'].values
        equal = np.array_equal(trend4, [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1])
        assert(equal)

    def test_get_target(self):
        data = {'close':[20, 21, 23, 19, 18, 24, 25, 26, 16, -10, -15, -18, -15, -8]}
        df = pd.DataFrame(data)
        df = findicators.add_technical_indicators(df, ["target"])
        target = df.loc[:,'target'].values
        equal = np.allclose(target, [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, np.nan], equal_nan=True)
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

    def test_vsa(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        df = findicators.add_technical_indicators(df, ['vsa'])

        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        #df.to_csv("./lib/data/test/findicators_vsa_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/findicators_vsa_reference.csv")

        for column in ["vsa_volume_1D","vsa_price_spread_1D","vsa_close_loc_1D","vsa_close_change_1D","vsa_volume_2D","vsa_price_spread_2D","vsa_close_loc_2D","vsa_close_change_2D","vsa_volume_3D","vsa_price_spread_3D","vsa_close_loc_3D","vsa_close_change_3D","vsa_volume_5D","vsa_price_spread_5D","vsa_close_loc_5D","vsa_close_change_5D","vsa_volume_20D","vsa_price_spread_20D","vsa_close_loc_20D","vsa_close_change_20D","vsa_volume_40D","vsa_price_spread_40D","vsa_close_loc_40D","vsa_close_change_40D","vsa_volume_60D","vsa_price_spread_60D","vsa_close_loc_60D","vsa_close_change_60D","outcomes_vsa"]:
            array = df[column].to_numpy()
            array_expected = expected_df[column].to_numpy()
            assert(np.allclose(array, array_expected))


    def test_temporal_indicators(self):
        idx = pd.Index(pd.date_range("19991231", periods=10), name='Date')
        df = pd.DataFrame([1]*10, columns=["Foobar"], index=idx)
        df = findicators.add_temporal_indicators(df, "Date")

        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/findicators_temporal_indicators_reference.csv")
        #assert_frame_equal(df, expected_df, check_dtype=False) // don't work :(
        assert(df.equals(expected_df))
