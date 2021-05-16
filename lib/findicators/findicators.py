import pandas as pd

# https://pythondata.com/stockstats-python-module-various-stock-market-statistics-indicators/
from stockstats import StockDataFrame as Sdf

# https://github.com/peerchemist/finta
from finta import TA


def remove_features(df, features):
    for feature in features:
        df.drop(feature, axis=1, inplace=True)
    return df

def add_technical_indicators(df, indicators):
    """
    calculate technical indicators
    use stockstats package to add technical indicators
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
    if 'Adj Close' in columns:
        df.rename(columns={'Adj Close': 'adj close'}, inplace=True)


    # call stockstats
    stock = Sdf.retype(df.copy())

    # add indicators to the dataframe
    if 'trend' in indicators:
        diff = df["close"] - df["close"].shift(1)
        df["trend"] = diff.gt(0).map({False: 0, True: 1})
        indicators.remove("trend")

    if 'macd' in indicators:
        df['macd'] = stock.get('macd').copy() # from stockstats
        #df['macd'] = TA.MACD(stock)['MACD'].copy() # from finta
        indicators.remove("macd")

    if 'ema' in indicators:
        df['ema'] = TA.EMA(stock).copy()
        indicators.remove("ema")

    if 'bbands' in indicators:
        bbands = TA.BBANDS(stock).copy()
        df = pd.concat([df, bbands], axis = 1)
        df.rename(columns={'BB_UPPER': 'bb_upper'}, inplace=True)
        df.rename(columns={'BB_MIDDLE': 'bb_middle'}, inplace=True)
        df.rename(columns={'BB_LOWER': 'bb_lower'}, inplace=True)
        indicators.remove("bbands")

    if 'rsi_30' in indicators:
        df['rsi_30'] = stock.get('rsi_30').copy()
        indicators.remove("rsi_30")
        
    if 'cci_30' in indicators:
        df['cci_30'] = stock.get('cci_30').copy()
        indicators.remove("cci_30")
        
    if 'dx_30' in indicators:
        df['dx_30'] = stock.get('dx_30').copy()
        indicators.remove("dx_30")

    if len(indicators) != 0:
        print("!!! add_technical_indicators !!! unknown indicators : {}".format(indicators))
    
    return df
