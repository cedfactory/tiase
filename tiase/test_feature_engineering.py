import pandas as pd
import numpy as np
from tiase.fimport import fimport
from tiase.findicators import findicators
from tiase.fdatapreprocessing import fdataprep
from tiase.featureengineering import fbalance,fprocessfeature
import pytest

class TestFeatureEngineering:

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

    def test_smote_balance(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        df = findicators.add_technical_indicators(df, ['simple_rtn','target','trend_1d'])
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
        val_counts = df['target'].value_counts()
        assert(val_counts[0] == 91)
        assert(val_counts[1] == 107)

        df = fbalance.balance_features(df, "smote")
        val_counts = df['target'].value_counts()
        assert(val_counts[0] == 107)
        assert(val_counts[1] == 107)

    def test_reductions(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        techindicators = ['simple_rtn', 'target', 'rsi_30', 'atr', 'williams_%r', 'macd', 'stoch_%k', 'stoch_%d', 'roc', 'mom', 'adx', 'er', 'cci_30', 'stc']
        df = findicators.add_technical_indicators(df, techindicators)
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
        df = df.astype({"target": int})

        for reduction in ["kbest_reduction", "correlation_reduction","pca_reduction","rfecv_reduction"]:
            df_reduction = fprocessfeature.process_features(df.copy(), [reduction])
            
            ref_file = "./tiase/data/test/featureengineering_"+reduction+"_reference.csv"
            #df_reduction.to_csv(ref_file)
            expected_df_reduction = fimport.get_dataframe_from_csv(ref_file)

            for column in df_reduction.columns:
                array = df_reduction[column].to_numpy()
                array_expected = expected_df_reduction[column].to_numpy()
                assert(np.allclose(array, array_expected))


    def test_reduction_vsa(self):
        df = self.get_real_dataframe()
        df = df.head(200)
        df = findicators.add_technical_indicators(df, ["vsa"])
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
        df_reduction = fprocessfeature.process_features(df.copy(), ["vsa_reduction"])

        ref_file = "./tiase/data/test/featureengineering_vsa_reduction_reference.csv"
        #df_reduction.to_csv(ref_file)
        expected_df_reduction = fimport.get_dataframe_from_csv(ref_file)

        for column in df_reduction.columns:
            array = df_reduction[column].to_numpy()
            array_expected = expected_df_reduction[column].to_numpy()
            assert(np.allclose(array, array_expected))

    def test_data_labeling(self):
        df = self.get_real_dataframe()
        df = df.head(150)
        df_labeling = fprocessfeature.process_features(df.copy(), ["data_labeling"])
        df_labeling = fdataprep.process_technical_indicators(df_labeling, ['missing_values']) # shit happens

        ref_file = "./tiase/data/test/featureengineering_data_labeling_reference.csv"
        #df_labeling.to_csv(ref_file)
        expected_df_labeling = fimport.get_dataframe_from_csv(ref_file)

        for column in df_labeling.columns:
            print(column)
            array_expected = expected_df_labeling[column].to_numpy()
            if array_expected.dtype != object:
                array = df_labeling[column].to_numpy(dtype = array_expected.dtype)
                assert(np.allclose(array, array_expected))

