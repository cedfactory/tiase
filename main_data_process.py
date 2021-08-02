from lib.fimport import *
from lib.findicators import *
from lib.featureengineering import fprocessfeature
from lib.fdatapreprocessing import fdataprep
import numpy as np

import pandas as pd
import os

from rich import print,inspect

def get_real_dataframe():
    filename = "./lib/data/test/google_stocks_data.csv"
    df = fimport.GetDataFrameFromCsv(filename)
    df = findicators.normalize_column_headings(df)
    return df

def get_synthetic_dataframe():
    y = synthetic.get_sinusoid(length=5, amplitude=1, frequency=.1, phi=0, height = 0)
    df = synthetic.create_dataframe(y, 0.)
    df = findicators.normalize_column_headings(df)
    return df


def check(action):
    print("check \"{}\"".format(action))

    if action == "missing_values":
        df = get_synthetic_dataframe()
        df["open"][1] = np.nan
        print(df.head())
        df = fdataprep.process_technical_indicators(df, ['missing_values'])
        print(df.head())
        
    elif action == "outliers_stdcutoff":
        df = get_synthetic_dataframe()
        print(df.head())
        df = fdataprep.process_technical_indicators(df, ['outliers_stdcutoff'])
        print(df.head())
        
    elif action == "discretization":
        df = get_real_dataframe()
        df = df.head(200)
        technical_indicators = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'cci_30', 'macd', 'stoch_%d', 'williams_%r', 'rsi_30']
        # todo : integrate ['wma', 'ema', 'sma']
        df = findicators.add_technical_indicators(df, technical_indicators)
        print(df.head())
        df = fdataprep.process_technical_indicators(df, ['discretization'])
        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])
        print(df.head())
        #df.to_csv("./lib/data/test/datapreprocess_discretization_reference.csv")

    elif action == "discretization_unsupervised":
        df = get_real_dataframe()
        df = df.head(200)
        technical_indicators = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'cci_30', 'macd', 'stoch_%d', 'williams_%r', 'rsi_30']
        # todo : integrate ['wma', 'ema', 'sma']
        df = findicators.add_technical_indicators(df, technical_indicators)
        print(df.head())
        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

        df = fdataprep.process_technical_indicators(df, ['discretization_unsupervised'])
        df = findicators.remove_features(df, ["high", "low", "open", "close", "adj_close", "volume"])
        print(df.head())
        #df.to_csv("./lib/data/test/datapreprocess_discretization_unsupervised_reference.csv")


#
# parse directory cac40
#
def cac40():
    directory = "./lib/data/CAC40/"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = fimport.GetDataFrameFromCsv(directory+"/"+filename)
            technical_indicators = ["trend_1d","macd","rsi_30","cci_30","williams_%r","stoch_%k","stoch_%d","er","stc"]
            technical_indicators.extend(["sma_5","sma_10","sma_20"])
            technical_indicators.extend(["ema_5","ema_10","ema_20"])
            technical_indicators.extend(["wma_5","wma_10","wma_20"])
            technical_indicators.extend(["atr","adx","roc"])
            technical_indicators.extend(["simple_rtn", "mom", "target"])

            df = findicators.add_technical_indicators(df, technical_indicators)
            trend_ratio, true_positive, true_negative, false_positive, false_negative = findicators.get_trend_info(df)
            value = filename[:len(filename)-4]
            print("{} ({}),{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(value, fimport.cac40[value], trend_ratio, true_positive, true_negative, false_positive, false_negative))

            df = fdataprep.process_technical_indicators(df, ['missing_values'])
            visu.DisplayHistogramFromDataframe(df, 'simple_rtn', './tmp/' + value + '_close.png')

            df_copy = df.copy()
            df = fdataprep.process_technical_indicators(df, ['outliers_winsorize'])
            visu.DisplayOutliersFromDataframe(df_copy, df, './tmp/' + value + '_outliers.png')

            df = fdataprep.process_technical_indicators(df, ['discretization_unsupervised'])
            df = fprocessfeature.process_features(df, ['correlation_reduction'])
            df = fprocessfeature.process_features(df, ['smote'])
            df = fprocessfeature.process_features(df, ['rfecv_reduction'])
            df = fdataprep.process_technical_indicators(df, ['discretization'])
            df = fprocessfeature.process_features(df, ['correlation_reduction'])
            df = fprocessfeature.process_features(df, ['pca_reduction'])
            df = fdataprep.process_technical_indicators(df, ['discretization'])
            df = fprocessfeature.process_features(df, ['correlation_reduction'])
            df = fdataprep.process_technical_indicators(df, ['discretization_unsupervised'])
            df = fdataprep.process_technical_indicators(df, ['transformation_log'])
            df = fdataprep.process_technical_indicators(df, ['transformation_x2'])
            df = fdataprep.process_technical_indicators(df, ['outliers_stdcutoff'])
            df = fdataprep.process_technical_indicators(df, ['outliers_ema'])
            df = fdataprep.process_technical_indicators(df, ['outliers_mam'])
            df = fdataprep.process_technical_indicators(df, ['outliers_winsorize'])

            break

_usage_str = """
Options:
    [--check --cac40]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check" and len(sys.argv) > 2: check(sys.argv[2])
        elif sys.argv[1] == "--cac40": cac40()
        else: _usage()
    else: _usage()
