import pandas as pd
import numpy as np
from lib.fimport import *
from lib.findicators import *
from lib.fdatapreprocessing import fdataprep
from lib.featureengineering import fbalance,fprocessfeature
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

    def test_smote_balance(self):
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

    def test_reductions(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        techindicators = ['rsi_30', 'atr', 'williams_%r', 'macd', 'stoch_%k', 'stoch_%d', 'roc', 'mom', 'adx', 'er', 'cci_30', 'stc', 'target']
        df = findicators.add_technical_indicators(df, techindicators)
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
        df = df.astype({"target": int})

        for reduction in ["correlation_reduction","pca_reduction","rfecv_reduction"]:
            df_reduction = fprocessfeature.process_features(df.copy(), [reduction])
            
            ref_file = "./lib/data/test/featureengineering_"+reduction+"_reference.csv"
            #df.to_csv(ref_file)
            expected_df_reduction = fimport.GetDataFrameFromCsv(ref_file)

            for column in df_reduction.columns:
                array = df_reduction[column].to_numpy()
                array_expected = expected_df_reduction[column].to_numpy()
                assert(np.allclose(array, array_expected))


