import pandas as pd
import numpy as np
from lib.fimport import *
from lib.findicators import *
from lib.featureengineering import *
import pytest

class TestFeatureEngineering:

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
        df = self.get_real_dataframe()
        df = df.head(200)
        df = findicators.add_technical_indicators(df, ['trend_1d'])
        df = df.astype({"trend_1d": int})
        val_counts = df['trend_1d'].value_counts()
        assert(val_counts[0] == 92)
        assert(val_counts[1] == 108)

        df = fbalance.smote_balance(df, 'trend_1d')
        val_counts = df['trend_1d'].value_counts()
        assert(val_counts[0] == 108)
        assert(val_counts[1] == 108)