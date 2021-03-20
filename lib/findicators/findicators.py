import pandas as pd
from stockstats import StockDataFrame as Sdf

def add_technical_indicators(df):
    """
    calculate technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """

    # rename columns if necessary
    columns = list(df.columns)
    if 'Open' in columns:
        df.rename(columns={'Open': 'open'}, inplace=True)
    if 'Close' in columns:
        df.rename(columns={'Close': 'close'}, inplace=True)
    if 'Low' in columns:
        df.rename(columns={'Low': 'low'}, inplace=True)
    if 'High' in columns:
        df.rename(columns={'High': 'high'}, inplace=True)
    if 'Volume' in columns:
        df.rename(columns={'Volume': 'volume'}, inplace=True)

    # call stockstats
    stock = Sdf.retype(df.copy())

    # add indicators to the dataframe
    df = df.join(stock.get('macd'))
    df = df.join(stock.get('rsi_30'))
    df = df.join(stock.get('cci_30'))
    df = df.join(stock.get('dx_30'))
    
    return df
