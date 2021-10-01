from tiar.fimport import fimport,synthetic
from tiar.findicators import findicators
from tiar.fdatapreprocessing import fdataprep
from tiar.featureengineering import fprocessfeature,fbalance
import numpy as np

import pandas as pd
import os

from rich import print,inspect

def get_real_dataframe():
    filename = "./tiar/data/test/google_stocks_data.csv"
    df = fimport.get_dataframe_from_csv(filename)
    df = findicators.normalize_column_headings(df)
    return df

def get_synthetic_dataframe():
    y = synthetic.get_sinusoid(length=5, amplitude=1, frequency=.1, phi=0, height = 0)
    df = synthetic.create_dataframe(y, 0.)
    df = findicators.normalize_column_headings(df)
    return df

def check(action):
    print("check \"{}\"".format(action))
    
    if action == "smote":
        df = get_real_dataframe()
        df = df.head(200)
        df = findicators.add_technical_indicators(df, ['simple_rtn','target','trend_1d'])
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
        print(df.head())
        print(df['target'].value_counts())

        df = fbalance.balance_features(df, "smote")
        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])
        print(df.head())
        print(df['target'].value_counts())
        #df.to_csv("./tiar/data/test/featureengineering_smote_reference.csv")

    elif action == "reductions":
        df = get_real_dataframe()
        df = df.head(200)
        techindicators = ['simple_rtn', 'target', 'rsi_30', 'atr', 'williams_%r', 'macd', 'stoch_%k', 'stoch_%d', 'roc', 'mom', 'adx', 'er', 'cci_30', 'stc']
        df = findicators.add_technical_indicators(df, techindicators)
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
        df = df.astype({"target": int})
        print(df.head())

        for reduction in ["kbest_reduction", "correlation_reduction","pca_reduction","rfecv_reduction"]:
            df_reduction = fprocessfeature.process_features(df.copy(), [reduction])
            print(df_reduction.head())
            df_reduction.to_csv("./tiar/data/test/featureengineering_"+reduction+"_reference.csv")

        # vsa_reduction
        df = get_real_dataframe()
        df = findicators.add_technical_indicators(df, "vsa")
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
        df_reduction = fprocessfeature.process_features(df.copy(), ["vsa_reduction"])
        print(df_reduction.head())
            
    elif action == "labeling":
        df = get_real_dataframe()
        #df = df.head(200)
        print()

        df_labeling = fprocessfeature.process_features(df.copy(), ["data_labeling"])
        print()

    else:
        print("action {} is unknown".format(action))


_usage_str = """
Options:
    [--check [ process ]]
process in [ smote,reductions,labeling ]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check" and len(sys.argv) > 2: check(sys.argv[2])
        else: _usage()
    else: _usage()
