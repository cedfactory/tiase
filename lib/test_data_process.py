import pandas as pd
import numpy as np
from lib.fimport import fimport,synthetic
from lib.findicators import findicators
from lib.fdatapreprocessing import fdataprep
import pytest

class TestDataProcess:

    def get_real_dataframe(self):
        filename = "./lib/data/test/google_stocks_data.csv"
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

    def test_discretization(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        technical_indicators = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'cci_30', 'macd', 'stoch_%d', 'williams_%r', 'rsi_30']
        df = findicators.add_technical_indicators(df, technical_indicators)

        df = fdataprep.process_technical_indicators(df, ['discretization_supervised'])
        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])

        #df.to_csv("./lib/data/test/datapreprocess_discretization_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/datapreprocess_discretization_reference.csv")
        assert(df.equals(expected_df))

    def test_discretization_unsupervised(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        technical_indicators = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'cci_30', 'macd', 'stoch_%d', 'williams_%r', 'rsi_30']
        df = findicators.add_technical_indicators(df, technical_indicators)
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        df = fdataprep.process_technical_indicators(df, ['discretization_unsupervised'])

        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])

        #df.to_csv("./lib/data/test/datapreprocess_discretization_unsupervised_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/datapreprocess_discretization_unsupervised_reference.csv")

        assert(df.equals(expected_df))


    def test_normalize_outliers_std_cutoff(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = fdataprep.process_technical_indicators(df, ['outliers_normalize_stdcutoff'])

        df = findicators.remove_features(df, ["high", "low", "open", "adj_close", "volume"])

        #df.to_csv("./lib/data/test/datapreprocess_normalize_outliers_std_cutoff_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/datapreprocess_normalize_outliers_std_cutoff_reference.csv")
        
        array = df['close'].to_numpy()
        array_expected = expected_df['close'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_cut_outliers_std_cutoff(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = fdataprep.process_technical_indicators(df, ['outliers_cut_stdcutoff'])

        df = findicators.remove_features(df, ["high", "low", "open", "adj_close", "volume"])

        #df.to_csv("./lib/data/test/datapreprocess_cut_outliers_std_cutoff_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/datapreprocess_cut_outliers_std_cutoff_reference.csv")
        
        array = df['close'].to_numpy()
        array_expected = expected_df['close'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_normalize_outliers_winsorize(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = fdataprep.process_technical_indicators(df, ['outliers_normalize_winsorize'])

        df = findicators.remove_features(df, ["high", "low", "open", "adj_close", "volume", "simple_rtn"])

        #df.to_csv("./lib/data/test/datapreprocess_normalize_outliers_winsorize_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/datapreprocess_normalize_outliers_winsorize_reference.csv")
        
        array = df['close'].to_numpy()
        array_expected = expected_df['close'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_normalize_outliers_mam(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = fdataprep.process_technical_indicators(df, ['outliers_mam'])

        df = findicators.remove_features(df, ["high", "low", "open", "adj_close", "volume"])

        #df.to_csv("./lib/data/test/datapreprocess_outliers_mam_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/datapreprocess_outliers_mam_reference.csv")
        
        array = df['close'].to_numpy()
        array_expected = expected_df['close'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_normalize_outliers_ema(self):
        df = self.get_real_dataframe()
        df = df.head(200)

        df = fdataprep.process_technical_indicators(df, ['outliers_ema'])

        df = findicators.remove_features(df, ["high", "low", "open", "adj_close", "volume"])

        #df.to_csv("./lib/data/test/datapreprocess_outliers_ema_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/datapreprocess_outliers_ema_reference.csv")
        
        array = df['close'].to_numpy()
        array_expected = expected_df['close'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_transformation_log(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        technical_indicators = ['simple_rtn']
        df = findicators.add_technical_indicators(df, technical_indicators)
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        df = fdataprep.process_technical_indicators(df, ['transformation_log'])

        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])

        #df.to_csv("./lib/data/test/datapreprocess_transformation_log_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/datapreprocess_transformation_log_reference.csv")
        
        array = df['simple_rtn'].to_numpy()
        array_expected = expected_df['simple_rtn'].to_numpy()
        assert(np.allclose(array, array_expected))

    def test_transformation_x2(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        technical_indicators = ['simple_rtn']
        df = findicators.add_technical_indicators(df, technical_indicators)
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        df = fdataprep.process_technical_indicators(df, ['transformation_x2'])

        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])

        #df.to_csv("./lib/data/test/datapreprocess_transformation_x2_reference.csv")
        expected_df = fimport.get_dataframe_from_csv("./lib/data/test/datapreprocess_transformation_x2_reference.csv")
        
        array = df['simple_rtn'].to_numpy()
        array_expected = expected_df['simple_rtn'].to_numpy()
        assert(np.allclose(array, array_expected))
