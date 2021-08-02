import pandas as pd
import numpy as np
from lib.fimport import *
from lib.findicators import *
from lib.fdatapreprocessing import *
import pytest

class TestDataProcess:

    def get_real_dataframe(self):
        filename = "./lib/data/test/google_stocks_data.csv"
        df = fimport.GetDataFrameFromCsv(filename)
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

        assert(df.shape[0] == 5)
        df = fdataprep.process_technical_indicators(df, ['missing_values'])
        assert(df.shape[0] == 4)

    def test_discretization(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        technical_indicators = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'cci_30', 'macd', 'stoch_%d', 'williams_%r', 'rsi_30']
        df = findicators.add_technical_indicators(df, technical_indicators)

        df = fdataprep.process_technical_indicators(df, ['discretization'])
        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])

        expected_df = fimport.GetDataFrameFromCsv("./lib/data/test/datapreprocess_discretization_reference.csv")
        assert(df.equals(expected_df))

    def test_discretization_unsupervised(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        technical_indicators = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'cci_30', 'macd', 'stoch_%d', 'williams_%r', 'rsi_30']
        df = findicators.add_technical_indicators(df, technical_indicators)
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        df = fdataprep.process_technical_indicators(df, ['discretization_unsupervised'])

        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])

        expected_df = fimport.GetDataFrameFromCsv("./lib/data/test/datapreprocess_discretization_unsupervised_reference.csv")
        assert(df.equals(expected_df))

