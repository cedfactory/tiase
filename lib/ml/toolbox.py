import pandas as pd
import numpy as np

def AddTrend(df, columnSource, columnTarget):
    diff = df[columnSource] - df[columnSource].shift(1)
    df[columnTarget] = diff.gt(0).map({False: 0, True: 1})
    return df
    
def get_mape(y_true, y_pred):
    """
    Compute Mean Absolute Percentage Error (MAPE)
    INPUT:
    y_true - actual variable
    y_pred - predicted variable
    OUTPUT:
    mape - Mean Absolute Percentage Error (%)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def get_rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE)
    INPUT:
    y_true - actual variable
    y_pred - predicted variable
    OUTPUT:
    rmse - Root Mean Squared Error
    """
    rmse = np.sqrt(np.mean(np.power((y_true - y_pred),2)))
    return rmse
