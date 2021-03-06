from tiase.fimport import fimport
from tiase.findicators import findicators
from tiase.fdatapreprocessing import fdataprep
from . import alfred
import pandas as pd
# todo : try to use from pandas._testing import assert_frame_equal
import numpy as np
import pytest
import os
import datetime

g_generate_references = False

class TestIndicators:
    
    def get_real_dataframe(self):
        filename = "./tiase/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)
        df = findicators.normalize_column_headings(df)
        return df

    def test_get_all_default_technical_indicators(self):
        ti = findicators.get_all_default_technical_indicators()
        assert(len(ti) == 29)
        expected_ti = ["trend_1d","macd","macds","macdh","bbands","rsi_30","cci_30","dx_30","williams_%r","stoch_%k","stoch_%d","er","stc","atr","adx","roc","mom","simple_rtn"]
        expected_ti.extend(["wma_5","wma_10","wma_15"])
        expected_ti.extend(["sma_5","sma_10","sma_15","sma_20"])
        expected_ti.extend(["ema_10","ema_20","ema_50"])
        expected_ti.extend(["labeling"])
        assert(ti == expected_ti)

    def test_number_colums(self):
        df = fimport.get_dataframe_from_csv("./tiase/data/CAC40/AI.PA.csv")
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

    def test_get_stats_for_trend_up(self):
        data = {'close':[20, 21, 23, 19, 18, 24, 25, 26, 27, 28]}
        df = pd.DataFrame(data)
        trend_ratio = findicators.get_stats_for_trend_up(df, 1)
        assert(trend_ratio == 70.)

    def test_get_stats_on_trend_today_equals_trend_tomorrow(self):
        data = {'close':[20, 21, 23, 19, 18, 24, 25, 26, 27, 28]}
        df = pd.DataFrame(data)
        true_positive, true_negative, false_positive, false_negative = findicators.get_stats_on_trend_today_equals_trend_tomorrow(df)
        assert(true_positive == pytest.approx(55.5555, 0.001))
        assert(true_negative == pytest.approx(11.111, 0.001))
        assert(false_positive == pytest.approx(11.111, 0.001))
        assert(false_negative == pytest.approx(22.222, 0.001))

    def test_vsa(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        df = findicators.add_technical_indicators(df, ['vsa'])

        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        #df.to_csv("./tiase/data/test/findicators_vsa_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/findicators_vsa_reference.csv")

        for column in ["vsa_volume_1D","vsa_price_spread_1D","vsa_close_loc_1D","vsa_close_change_1D","vsa_volume_2D","vsa_price_spread_2D","vsa_close_loc_2D","vsa_close_change_2D","vsa_volume_3D","vsa_price_spread_3D","vsa_close_loc_3D","vsa_close_change_3D","vsa_volume_5D","vsa_price_spread_5D","vsa_close_loc_5D","vsa_close_change_5D","vsa_volume_20D","vsa_price_spread_20D","vsa_close_loc_20D","vsa_close_change_20D","vsa_volume_40D","vsa_price_spread_40D","vsa_close_loc_40D","vsa_close_change_40D","vsa_volume_60D","vsa_price_spread_60D","vsa_close_loc_60D","vsa_close_change_60D","outcomes_vsa"]:
            array = df[column].to_numpy()
            array_expected = expected_df[column].to_numpy()
            assert(np.allclose(array, array_expected))

    def labeling_common(self, dict_params, ref_csvfile, ref_barriers_csvfile):
        df = self.get_real_dataframe()
        df = df.head(150)
        df_labeling = findicators.add_technical_indicators(df, ['labeling'], dict_params)
        df_labeling = fdataprep.process_technical_indicators(df_labeling, ['missing_values']) # shit happens
        df_labeling = findicators.remove_features(df_labeling, ['high', 'low', 'open', 'volume', 'adj_close'])

        # final result
        if g_generate_references:
            df_labeling.to_csv(ref_csvfile)
        expected_df_labeling = fimport.get_dataframe_from_csv(ref_csvfile)

        for column in df_labeling.columns:
            array_expected = expected_df_labeling[column].to_numpy()
            if array_expected.dtype != object:
                array = df_labeling[column].to_numpy(dtype = array_expected.dtype)
                assert(np.allclose(array, array_expected))

        # barriers for debug
        gen_file = "./tmp/labeling_barriers.csv"
        if g_generate_references:
            os.rename(gen_file, ref_barriers_csvfile)
        ref_df_barriers = fimport.get_dataframe_from_csv(ref_barriers_csvfile)
        gen_df_barriers = fimport.get_dataframe_from_csv(gen_file)

        for column in gen_df_barriers.columns:
            ref_array = ref_df_barriers[column].to_numpy()
            if ref_array.dtype != object:
                gen_array = gen_df_barriers[column].to_numpy(dtype = ref_array.dtype)
                assert(np.allclose(gen_array, ref_array))

    def test_labeling_close_unbalanced(self):
        dict_params = {'labeling_debug':True, 'labeling_t_final':10, 'labeling_upper_multiplier':"2.", 'labeling_lower_multiplier':"2."}
        ref_csvfile = "./tiase/data/test/findicators_data_labeling_reference.csv"
        ref_barriers_csvfile = "./tiase/data/test/findicators_data_labeling_barriers_reference.csv"
        self.labeling_common(dict_params, ref_csvfile, ref_barriers_csvfile)

    def test_labeling_high_low_unbalanced(self):
        dict_params = {'labeling_debug':True, 'use_high_low':'1', 'labeling_t_final':10, 'labeling_upper_multiplier':"2.", 'labeling_lower_multiplier':"2."}
        ref_csvfile = "./tiase/data/test/findicators_data_labeling_high_low_reference.csv"
        ref_barriers_csvfile = "./tiase/data/test/findicators_data_labeling_high_low_barriers_reference.csv"
        self.labeling_common(dict_params, ref_csvfile, ref_barriers_csvfile)

    def test_labeling_close_balanced(self):
        dict_params = {'labeling_debug':True, "use_balanced_upper_multiplier":1, 'labeling_t_final':10, 'labeling_upper_multiplier':"2.", 'labeling_lower_multiplier':"2.",  "labeling_label_below":"0", "labeling_label_middle":"0", "labeling_label_above":"1"}
        ref_csvfile = "./tiase/data/test/findicators_data_labeling_close_balanced_reference.csv"
        ref_barriers_csvfile = "./tiase/data/test/findicators_data_labeling_close_balanced_barriers_reference.csv"
        self.labeling_common(dict_params, ref_csvfile, ref_barriers_csvfile)

    def test_labeling_high_low_balanced(self):
        dict_params = {'labeling_debug':True, 'use_high_low':'1', "use_balanced_upper_multiplier":1, 'labeling_t_final':10, 'labeling_upper_multiplier':"2.", 'labeling_lower_multiplier':"2.",  "labeling_label_below":"0", "labeling_label_middle":"0", "labeling_label_above":"1"}
        ref_csvfile = "./tiase/data/test/findicators_data_labeling_high_low_balanced_reference.csv"
        ref_barriers_csvfile = "./tiase/data/test/findicators_data_labeling_high_low_balanced_barriers_reference.csv"
        self.labeling_common(dict_params, ref_csvfile, ref_barriers_csvfile)


    def test_labeling_with_alfred(self):
        alfred.execute("./tiase/data/test/findicators_alfred_data_labeling.xml")
        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = findicators.remove_features(df_generated, ["high", "low", "open", "adj_close", "volume"])
        df_generated = df_generated.head(137) # ref has been computed with 150 first values & t_final=10
        df_generated["labeling"] = df_generated["labeling"].astype(int)

        ref_file = "./tiase/data/test/findicators_data_labeling_reference.csv"
        if g_generate_references:
            df_generated.to_csv(ref_file)
        df_expected = fimport.get_dataframe_from_csv(ref_file)

        assert(df_generated.equals(df_expected))

    def test_shift(self):
        data = {'close':[20., 21., 23., 19., 18., 24., 25., 26., 16.]}
        df = pd.DataFrame(data)

        df = findicators.shift(df, "close", 2)

        array = df.loc[:,'close'].values
        assert(np.array_equal(array, [np.NaN, np.NaN, 20., 21., 23., 19., 18., 24., 25.], equal_nan=True))

    def test_shift_with_alfred(self):
        alfred.execute("./tiase/data/test/alfred_shift.xml")
        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = findicators.remove_features(df_generated, ["high", "low", "open", "adj_close", "volume"])
        df_generated = df_generated.head(50)

        ref_file = "./tiase/data/test/findicators_shift_reference.csv"
        if g_generate_references:
            df_generated.to_csv(ref_file)
        df_expected = fimport.get_dataframe_from_csv(ref_file)

        assert(df_generated.equals(df_expected))

    def test_temporal_indicators(self):
        idx = pd.Index(pd.date_range("19991231", periods=10), name='Date')
        df = pd.DataFrame([1]*10, columns=["Foobar"], index=idx)
        df = findicators.add_temporal_indicators(df, "Date")

        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/findicators_temporal_indicators_reference.csv")
        #assert_frame_equal(df, expected_df, check_dtype=False) // don't work :(
        assert(df.equals(expected_df))
