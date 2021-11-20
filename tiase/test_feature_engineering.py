import pandas as pd
import numpy as np
import os
from tiase.fimport import fimport
from tiase.findicators import findicators
from tiase.fdatapreprocessing import fdataprep
from tiase.featureengineering import fbalance,fprocessfeature
from . import alfred
import pytest

g_generate_references = False

class TestFeatureEngineering:

    def get_real_dataframe(self):
        filename = "./tiase/data/test/google_stocks_data.csv"
        df = fimport.get_dataframe_from_csv(filename)
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
            if g_generate_references:
                df_reduction.to_csv(ref_file)
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
        if g_generate_references:
            df_reduction.to_csv(ref_file)
        expected_df_reduction = fimport.get_dataframe_from_csv(ref_file)

        for column in df_reduction.columns:
            array = df_reduction[column].to_numpy()
            array_expected = expected_df_reduction[column].to_numpy()
            assert(np.allclose(array, array_expected))

    def test_data_labeling(self):
        df = self.get_real_dataframe()
        df = df.head(150)
        df_labeling = fprocessfeature.process_features(df.copy(), ["data_labeling"], {'debug':True, 't_final':10, 'target_name':'target'})
        df_labeling = fdataprep.process_technical_indicators(df_labeling, ['missing_values']) # shit happens
        df_labeling = findicators.remove_features(df_labeling, ['high', 'low', 'open', 'volume', 'adj_close'])

        # final result
        ref_file = "./tiase/data/test/featureengineering_data_labeling_reference.csv"
        if g_generate_references:
            df_labeling.to_csv(ref_file)
        expected_df_labeling = fimport.get_dataframe_from_csv(ref_file)

        for column in df_labeling.columns:
            array_expected = expected_df_labeling[column].to_numpy()
            if array_expected.dtype != object:
                array = df_labeling[column].to_numpy(dtype = array_expected.dtype)
                assert(np.allclose(array, array_expected))

        # barriers for debug
        gen_file = "./tmp/labeling_barriers.csv"
        ref_file = "./tiase/data/test/featureengineering_data_labeling_barriers_reference.csv"
        if g_generate_references:
            os.rename(gen_file, ref_file)
        ref_df_barriers = fimport.get_dataframe_from_csv(ref_file)
        gen_df_barriers = fimport.get_dataframe_from_csv(gen_file)

        for column in gen_df_barriers.columns:
            ref_array = ref_df_barriers[column].to_numpy()
            if ref_array.dtype != object:
                gen_array = gen_df_barriers[column].to_numpy(dtype = ref_array.dtype)
                assert(np.allclose(gen_array, ref_array))


    def test_data_labeling_with_alfred(self):
        alfred.execute("./tiase/data/test/featureengineering_alfred_data_labeling.xml")
        df_generated = fimport.get_dataframe_from_csv("./tmp/out.csv")
        df_generated = findicators.remove_features(df_generated, ["high", "low", "open", "adj_close", "volume"])
        df_generated = df_generated.head(137) # ref has been computed with 150 first values & t_final=10
        df_generated["target"] = df_generated["target"].astype(int)

        if g_generate_references:
            df_generated.to_csv("./tiase/data/test/featureengineering_data_labeling_reference.csv")
        df_expected = fimport.get_dataframe_from_csv("./tiase/data/test/featureengineering_data_labeling_reference.csv")

        assert(df_generated.equals(df_expected))
