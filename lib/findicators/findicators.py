from parse import parse

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

    # Change all column headings to be lower case, and remove spacing
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

    # call stockstats
    stock = Sdf.retype(df.copy())

    # add indicators to the dataframe
    for indicator in indicators:

        result = parse('trend_{}d', indicator)
        if result != None and result[0].isdigit():
            seq = int(result[0])
            diff = df["close"] - df["close"].shift(1)
            trend_1d = diff.gt(0).map({False: 0, True: 1})
            df["trend_"+result[0]+"d"] = trend_1d.rolling(seq).mean()

        elif indicator == 'macd':
            df['macd'] = stock.get('macd').copy() # from stockstats
            #df['macd'] = TA.MACD(stock)['MACD'].copy() # from finta

        elif indicator == 'ema':
            df['ema'] = TA.EMA(stock).copy()

        elif indicator == 'bbands':
            bbands = TA.BBANDS(stock).copy()
            df = pd.concat([df, bbands], axis = 1)
            df.rename(columns={'BB_UPPER': 'bb_upper'}, inplace=True)
            df.rename(columns={'BB_MIDDLE': 'bb_middle'}, inplace=True)
            df.rename(columns={'BB_LOWER': 'bb_lower'}, inplace=True)

        elif indicator == 'rsi_30':
            df['rsi_30'] = stock.get('rsi_30').copy()
        
        elif indicator == 'cci_30':
            df['cci_30'] = stock.get('cci_30').copy()
        
        elif indicator == 'dx_30':
            df['dx_30'] = stock.get('dx_30').copy()

        else:
            print("!!! add_technical_indicators !!! unknown indicator : {}".format(indicator))
    
    return df
