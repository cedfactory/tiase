import pandas as pd
import numpy as np
from tiase.fimport import fimport,synthetic,visu
from tiase.findicators import findicators
from tiase.fdatapreprocessing import fdataprep
from . import alfred
import pytest

g_generate_references = False

class TestDataProcess:

    def get_real_dataframe(self):
        filename = "./tiase/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)
        df = findicators.normalize_column_headings(df)
        return df

    def get_synthetic_dataframe(self):
        y = synthetic.get_sinusoid(length=5, amplitude=1, frequency=.1, phi=0, height = 0)
        df = synthetic.create_dataframe(y, 0.)
        df = findicators.normalize_column_headings(df)
        return df

    def test_missing_values(self):
        df = self.get_synthetic_dataframe()
        df["open"][1] = np.nan
        df["close"][2] = np.inf
        df["high"][3] = -np.inf

        assert(df.shape[0] == 5)
        df = fdataprep.process_technical_indicators(df, ['missing_values'])
        assert(df.shape[0] == 2)

    def test_duplicates(self):
        data = {'A':[20, 21, 13, 21, 18], 'B':[18, 19, 23, 19, 17]}
        df = pd.DataFrame(data)

        assert(df.shape[0] == 5)
        df = fdataprep.process_technical_indicators(df, ['duplicates'])
        assert(df.shape[0] == 4)

    def test_discretization(self):
        df = self.get_real_dataframe()
        technical_indicators = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'cci_30', 'macd', 'stoch_%d', 'williams_%r', 'rsi_30', 'sma_9', 'ema_9', 'wma_9']
        technical_indicators_to_discretize = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'cci_30', 'macd', 'stoch_%d', 'williams_%r', 'rsi_30', 'sma', 'ema', 'wma']
        df = findicators.add_technical_indicators(df, technical_indicators)
        df = fdataprep.process_technical_indicators(df, ['missing_values'])

        df = fdataprep.process_technical_indicators(df, ['discretization_supervised'], technical_indicators_to_discretize)

        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])
        df = df.head(200)

        ref_file = "./tiase/data/test/datapreprocess_discretization_reference.csv"
        if g_generate_references:
            df.to_csv(ref_file)
        expected_df = fimport.get_dataframe_from_csv(ref_file)
        assert(df.equals(expected_df))

    def test_discretization_with_alfred(self):
        alfred.execute("./tiase/data/test/datapreprocess_alfred_discretization_supervised.xml")
        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = findicators.remove_features(df_generated, ["high", "low", "open", "close", "adj_close", "volume", "target"])
        df_generated = df_generated.head(200)

        ref_file = "./tiase/data/test/datapreprocess_discretization_reference.csv"
        if g_generate_references:
            df_generated.to_csv(ref_file)
        df_expected = fimport.get_dataframe_from_csv(ref_file)

        assert(df_generated.equals(df_expected))

    def test_discretization_unsupervised(self):
        df = self.get_real_dataframe()
        technical_indicators = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'cci_30', 'macd', 'stoch_%d', 'williams_%r', 'rsi_30']
        df = findicators.add_technical_indicators(df, technical_indicators)
        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        df = fdataprep.process_technical_indicators(df, ['discretization_unsupervised'], technical_indicators)
        
        df = df.head(200)

        if g_generate_references:
            df.to_csv("./tiase/data/test/datapreprocess_discretization_unsupervised_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/datapreprocess_discretization_unsupervised_reference.csv")

        assert(df.equals(expected_df))

    def test_discretization_unsupervised_with_alfred(self):
        alfred.execute("./tiase/data/test/datapreprocess_alfred_discretization_unsupervised.xml")
        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = findicators.remove_features(df_generated, ["high", "low", "open", "close", "adj_close", "volume", "target"])
        df_generated = df_generated.head(200)

        if g_generate_references:
            df_generated.to_csv("./tiase/data/test/datapreprocess_discretization_unsupervised_reference.csv")
        df_expected = fimport.get_dataframe_from_csv("./tiase/data/test/datapreprocess_discretization_unsupervised_reference.csv")

        assert(df_generated.equals(df_expected))

    #
    # Outliers
    #

    def test_normalize_outliers_std_cutoff(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = findicators.add_technical_indicators(df, ['simple_rtn'])
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
        df_original = df.copy()
        df = fdataprep.process_technical_indicators(df, ['outliers_normalize_stdcutoff'], ['simple_rtn'])
        df = findicators.remove_features(df, ['high', 'low', 'open', 'close', 'adj_close', 'volume'])

        # debug
        visu.display_outliers_from_dataframe(df_original, df, 'simple_rtn', './tmp/test_normalize_outliers_std_cutoff.png')

        if g_generate_references:
            df.to_csv("./tiase/data/test/datapreprocess_normalize_outliers_std_cutoff_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/datapreprocess_normalize_outliers_std_cutoff_reference.csv")

        array = df['simple_rtn'].to_numpy()
        array_expected = expected_df['simple_rtn'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_normalize_outliers_std_cutoff_and_update_close(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = findicators.add_technical_indicators(df, ['simple_rtn'])
        df = fdataprep.process_technical_indicators(df, ['outliers_normalize_stdcutoff'], ['simple_rtn'])

        # update close
        df_tmp = pd.DataFrame(df['close'].copy())
        df_tmp['close'] = df['close'][0] * (1. + df['simple_rtn']).cumprod()
        df_tmp['close'][0] = df['close'][0]
        df['close'] = df_tmp['close']

        df = findicators.remove_features(df, ['simple_rtn', 'high', 'low', 'open', 'adj_close', 'volume'])

        if g_generate_references:
            df.to_csv("./tiase/data/test/datapreprocess_normalize_outliers_std_cutoff_and_update_close_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/datapreprocess_normalize_outliers_std_cutoff_and_update_close_reference.csv")
        
        array = df['close'].to_numpy()
        array_expected = expected_df['close'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_cut_outliers_std_cutoff(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = findicators.add_technical_indicators(df, ['simple_rtn'])
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])
        df_original = df.copy()

        df = fdataprep.process_technical_indicators(df, ['outliers_cut_stdcutoff'], ['simple_rtn'])

        # debug
        visu.display_outliers_from_dataframe(df_original, df, 'simple_rtn', './tmp/test_cut_outliers_std_cutoff.png')

        if g_generate_references:
            df.to_csv("./tiase/data/test/datapreprocess_cut_outliers_std_cutoff_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/datapreprocess_cut_outliers_std_cutoff_reference.csv")
        
        array = df['simple_rtn'].to_numpy()
        array_expected = expected_df['simple_rtn'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_normalize_outliers_winsorize(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = findicators.add_technical_indicators(df, ['simple_rtn'])
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
        df_original = df.copy()
        df = fdataprep.process_technical_indicators(df, ['outliers_normalize_winsorize'], ['simple_rtn'])
        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])

        # debug
        visu.display_outliers_from_dataframe(df_original, df, 'simple_rtn', './tmp/test_normalize_outliers_winsorize.png')

        if g_generate_references:
            df.to_csv("./tiase/data/test/datapreprocess_normalize_outliers_winsorize_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/datapreprocess_normalize_outliers_winsorize_reference.csv")
        
        array = df['simple_rtn'].to_numpy()
        array_expected = expected_df['simple_rtn'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_normalize_outliers_normalize_mam(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = fdataprep.process_technical_indicators(df, ['outliers_normalize_mam'])
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        df = findicators.remove_features(df, ["high", "low", "open", "adj_close", "volume"])

        if g_generate_references:
            df.to_csv("./tiase/data/test/datapreprocess_outliers_mam_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/datapreprocess_outliers_mam_reference.csv")
        
        array = df['close'].to_numpy()
        array_expected = expected_df['close'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_normalize_outliers_normalize_ema(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = fdataprep.process_technical_indicators(df, ['outliers_normalize_ema'])
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        df = findicators.remove_features(df, ["high", "low", "open", "adj_close", "volume"])

        if g_generate_references:
            df.to_csv("./tiase/data/test/datapreprocess_outliers_ema_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/datapreprocess_outliers_ema_reference.csv")
        
        array = df['close'].to_numpy()
        array_expected = expected_df['close'].to_numpy()
        assert(np.allclose(array, array_expected))

    #
    # Transformations
    # 

    def test_transformation_log(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        technical_indicators = ['simple_rtn']
        df = findicators.add_technical_indicators(df, technical_indicators)
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        df = fdataprep.process_technical_indicators(df, ['transformation_log'], technical_indicators)

        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])

        if g_generate_references:
            df.to_csv("./tiase/data/test/datapreprocess_transformation_log_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/datapreprocess_transformation_log_reference.csv")
        
        array = df['simple_rtn'].to_numpy()
        array_expected = expected_df['simple_rtn'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_transformation_x2(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        technical_indicators = ['simple_rtn']
        df = findicators.add_technical_indicators(df, technical_indicators)
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        df = fdataprep.process_technical_indicators(df, ['transformation_x2'], technical_indicators)

        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])

        if g_generate_references:
            df.to_csv("./tiase/data/test/datapreprocess_transformation_x2_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./tiase/data/test/datapreprocess_transformation_x2_reference.csv")
        
        array = df['simple_rtn'].to_numpy()
        array_expected = expected_df['simple_rtn'].to_numpy()
        assert(np.allclose(array, array_expected))
