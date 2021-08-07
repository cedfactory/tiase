import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from ..featureengineering import fselection


def data_discretization(df, columns):
    for col in columns:
        if col == "rsi_30":
            d1 = pd.DataFrame(df[col])
            d1["rsi_t"] = d1[col].copy()
            d1["rsi_t-1"] = d1["rsi_t"].shift(1)

            condition1 = (d1['rsi_t'] < 30) | (d1['rsi_t'] > d1['rsi_t-1'])
            condition2 = (d1['rsi_t'] > 70) | (d1['rsi_t'] < d1['rsi_t-1'])

            d1['dis_' + col] = np.where(condition1, 1, d1['rsi_t'])
            d1['dis_' + col] = np.where((d1['dis_' + col] != 1) | condition2, 0, d1['dis_' + col])
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "roc":
            d1 = pd.DataFrame(df[col])
            d1["roc_t"] = d1[col].copy()
            d1["roc_t-1"] = d1["roc_t"].shift(1)

            condition1 = (d1['roc_t'] < 0) | (d1['roc_t'] > d1['roc_t-1'])
            condition2 = (d1['roc_t'] > 0) | (d1['roc_t'] < d1['roc_t-1'])

            d1['dis_' + col] = np.where(condition1, 1, d1['roc_t'])
            d1['dis_' + col] = np.where((d1['dis_' + col] != 1) | condition2, 0, d1['dis_' + col])
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "adx":
            d1 = pd.DataFrame(df[col])
            d1["adx_t"] = d1[col].copy()
            d1["adx_t-1"] = d1["adx_t"].shift(1)

            condition1 = (d1['adx_t'] < 30) | (d1['adx_t'] > d1['adx_t-1'])
            condition2 = (d1['adx_t'] > 70) | (d1['adx_t'] < d1['adx_t-1'])

            d1['dis_' + col] = np.where(condition1, 1, d1['adx_t'])
            d1['dis_' + col] = np.where((d1['dis_' + col] != 1) | condition2, 0, d1['dis_' + col])
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "stc":
            d1 = pd.DataFrame(df[col])
            d1["stc_t"] = d1[col].copy()
            d1["stc_t-1"] = d1["stc_t"].shift(1)

            condition1 = (d1['stc_t'] < 25) | (d1['stc_t'] > d1['stc_t-1'])
            condition2 = (d1['stc_t'] > 75) | (d1['stc_t'] < d1['stc_t-1'])

            d1['dis_' + col] = np.where(condition1, 1, d1['stc_t'])
            d1['dis_' + col] = np.where((d1['dis_' + col] != 1) | condition2, 0, d1['dis_' + col])
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "cci_30":
            d1 = pd.DataFrame(df[col])
            d1["cci_t"] = d1[col].copy()
            d1["cci_t-1"] = d1["cci_t"].shift(1)

            condition1 = (d1['cci_t'] < -200) | (d1['cci_t'] > d1['cci_t-1'])
            condition2 = (d1['cci_t'] > 200) | (d1['cci_t'] < d1['cci_t-1'])

            d1['dis_' + col] = np.where(condition1, 1, d1['cci_t'])
            d1['dis_' + col] = np.where((d1['dis_' + col] != 1) | condition2, 0, d1['dis_' + col])
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "atr":
            d1 = pd.DataFrame(df[col])
            d1["atr_t"] = d1[col].copy()
            d1["atr_t-1"] = d1["atr_t"].shift(1)

            condition = (d1['atr_t'] > d1['atr_t-1'])
            d1['dis_' + col] = np.where(condition, 1, 0)
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "williams_%r":
            d1 = pd.DataFrame(df[col])
            d1["wr_t"] = d1[col].copy()
            d1["wr_t-1"] = d1["wr_t"].shift(1)

            condition = (d1['wr_t'] > d1['wr_t-1'])
            d1['dis_' + col] = np.where(condition, 1, 0)
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "stoch_%d":
            d1 = pd.DataFrame(df[col])
            d1["d_t"] = d1[col].copy()
            d1["d_t-1"] = d1["d_t"].shift(1)

            condition = (d1['d_t'] > d1['d_t-1'])
            d1['dis_' + col] = np.where(condition, 1, 0)
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "stoch_%k":
            d1 = pd.DataFrame(df[col])
            d1["k_t"] = d1[col].copy()
            d1["k_t-1"] = d1["k_t"].shift(1)

            condition = (d1['k_t'] > d1['k_t-1'])
            d1['dis_' + col] = np.where(condition, 1, 0)
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "er":
            d1 = pd.DataFrame(df[col])
            d1["er_t"] = d1[col].copy()
            d1["er_t-1"] = d1["er_t"].shift(1)

            condition = (d1['er_t'] > d1['er_t-1'])
            d1['dis_' + col] = np.where(condition, 1, 0)
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "macd":
            d1 = pd.DataFrame(df[col])
            d1["macd_t"] = d1[col].copy()
            d1["macd_t-1"] = d1["macd_t"].shift(1)

            condition = (d1['macd_t'] > d1['macd_t-1'])
            d1['dis_' + col] = np.where(condition, 1, 0)
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "mom":
            d1 = pd.DataFrame(df[col])
            d1["mom_t"] = d1[col].copy()
            d1["mom_t-1"] = d1["mom_t"].shift(1)

            condition = (d1['mom_t'] > d1['mom_t-1'])
            d1['dis_' + col] = np.where(condition, 1, 0)
            df[col] = d1["dis_" + col].copy().astype(int)

        if col == "sma":
            col_sma_lst = [item for item in df.columns if item.startswith(col)]
            d1 = pd.DataFrame(df[col_sma_lst])
            d1["close_t"] = df["close"].copy()
            for col_sma in col_sma_lst:
                d1[col_sma + "_t"] = d1[col_sma].copy()
                condition = (d1["close_t"] > d1[col_sma + "_t"])
                d1["dis_" + col_sma] = np.where(condition, 1, 0)
                df[col_sma] = d1["dis_" + col_sma].copy().astype(int)

        if col == "ema":
            col_ema_lst = [item for item in df.columns if item.startswith(col)]
            d1 = pd.DataFrame(df[col_ema_lst])
            d1["close_t"] = df["close"].copy()
            for col_ema in col_ema_lst:
                d1[col_ema + "_t"] = d1[col_ema].copy()
                condition = (d1["close_t"] > d1[col_ema + "_t"])
                d1["dis_" + col_ema] = np.where(condition, 1, 0)
                df[col_ema] = d1["dis_" + col_ema].copy().astype(int)

        if col == "wma":
            col_wma_lst = [item for item in df.columns if item.startswith(col)]
            d1 = pd.DataFrame(df[col_wma_lst])
            d1["close_t"] = df["close"].copy()
            for col_wma in col_wma_lst:
                d1[col_wma + "_t"] = d1[col_wma].copy()
                condition = (d1["close_t"] > d1[col_wma + "_t"])
                d1["dis_" + col_wma] = np.where(condition, 1, 0)
                df[col_wma] = d1["dis_" + col_wma].copy().astype(int)

    return df


def data_discretization_unsupervized(df, columns, nb_bins, strategy):
    columns = fselection.get_sma_ema_wma(df, columns)

    d1 = df[columns].copy()
    d1_index = d1.index.tolist()

    kbins = KBinsDiscretizer(n_bins=nb_bins, encode='ordinal', strategy=strategy)
    data_trans = kbins.fit_transform(d1)

    d1 = pd.DataFrame(data=data_trans, columns=columns, index=d1_index)

    for column in columns:
        df[column] = d1[column].copy()

    return df
