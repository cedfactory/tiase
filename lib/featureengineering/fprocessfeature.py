
from . import fselection,fbalance

def process_features(df, featureengineering):
    # process data indicators
    for process in featureengineering:
        if process == 'correlation_reduction':
            # columns = df.columns
            columns = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'wma', 'ema', 'sma', 'cci_30', 'macd',
                       'stoch_%d', 'williams_%r', 'rsi_30']
            df = fselection.correlation_reduction(df, columns)
        elif process == 'pca_reduction':
            # columns = df.columns
            columns = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'wma', 'ema', 'sma', 'cci_30', 'macd',
                       'stoch_%d', 'williams_%r', 'rsi_30']
            df = fselection.pca_reduction(df, columns)
        elif process == 'rfecv_reduction':
            # columns = df.columns
            columns = ['atr', 'mom', 'roc', 'er', 'adx', 'stc', 'stoch_%k', 'wma', 'ema', 'sma', 'cci_30', 'macd',
                       'stoch_%d', 'williams_%r', 'rsi_30']
            df = fselection.rfecv_reduction(df, columns)
        elif process == 'smote':
            df = fbalance.smote_balance(df)
        elif process == 'vsa_reduction':
            df = fselection.vsa_corr_selection(df)
        else:
            print(":fix: process {} is unknown".format(process))

    return df
