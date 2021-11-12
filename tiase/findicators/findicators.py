from parse import parse
import numpy as np

import pandas as pd

from . import vsa

# https://pythondata.com/stockstats-python-module-various-stock-market-statistics-indicators/
from stockstats import StockDataFrame as Sdf

# https://github.com/peerchemist/finta
from finta import TA

def normalize_column_headings(df):
    # Change all column headings to be lower case, and remove spacing
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
    return df

def remove_features(df, features):
    for feature in features:
        try:
            df.drop(feature, axis=1, inplace=True)
        except KeyError as feature:
            print("{}. Columns are {}".format(feature, df.columns))
    return df

def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)

def add_temporal_indicators(df, field_name, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."

    # Change all column headings to be lower case, and remove spacing
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

    if field_name not in df.columns and field_name != df.index.name:
        print("[add_temporal_indicators] {} is not present among the columns {} or in the index {}".format(datefield, df.columns, df.index.name))
        return df

    # if the datefield is the index of the dataframe, we create a temporary column
    field_to_drop = False
    if field_name == df.index.name:
        field_name = 'DateTmp'
        df[field_name] = df.index
        field_to_drop = True

    make_date(df, field_name)

    field = df[field_name]
    prefix = "" #ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower()) if n != 'Week' else week
    mask = ~field.isna()
    df[prefix + 'Elapsed'] = np.where(mask, field.values.astype(np.int64) // 10 ** 9, np.nan)
    if field_to_drop: df.drop(field_name, axis=1, inplace=True)

    return df

def get_all_default_technical_indicators():
    technical_indicators = ["trend_1d","macd","macds","macdh","bbands","rsi_30","cci_30","dx_30","williams_%r","stoch_%k","stoch_%d","er","stc","atr","adx","roc","mom","simple_rtn"]
    technical_indicators.extend(["wma_5","wma_10","wma_15"])
    technical_indicators.extend(["sma_5","sma_10","sma_15","sma_20"])
    technical_indicators.extend(["ema_10","ema_20","ema_50"])
    return technical_indicators

def add_technical_indicators(df, indicators):
    """
    calculate technical indicators
    use stockstats package to add technical indicators
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """

    df = normalize_column_headings(df)

    # call stockstats
    stock = Sdf.retype(df.copy())

    # add indicators to the dataframe
    for indicator in indicators:

        trend_parsed = parse('trend_{}d', indicator)
        sma_parsed = parse('sma_{}', indicator)
        ema_parsed = parse('ema_{}', indicator)
        wma_parsed = parse('wma_{}', indicator)

        if trend_parsed != None and trend_parsed[0].isdigit():
            seq = int(trend_parsed[0])
            diff = df["close"] - df["close"].shift(seq)
            df["trend_"+str(seq)+"d"] = diff.gt(0).map({False: 0, True: 1})

        elif sma_parsed != None and sma_parsed[0].isdigit():
            seq = int(sma_parsed[0])
            df["sma_"+str(seq)] = TA.SMA(stock, seq).copy()

        elif ema_parsed != None and ema_parsed[0].isdigit():
            period = int(ema_parsed[0])
            df["ema_"+str(period)] = TA.EMA(stock, period = period).copy()

        elif wma_parsed != None and wma_parsed[0].isdigit():
            period = int(wma_parsed[0])
            df["wma_"+str(period)] = TA.WMA(stock, period = period).copy()

        elif indicator == "on_balance_volume":
            # ref : https://medium.com/analytics-vidhya/analysis-of-stock-price-predictions-using-lstm-models-f993faa524c4

            if "volume" not in df.columns or ("adj_close" not in df.columns and "close" not in df.columns):
                print("!!! add_technical_indicators !!! on_balance_volume indicator : can't be evaluated")
                return df

            new_balance_volume = [0]
            tally = 0
            adj_close = "adj_close"
            if adj_close not in df.columns:
                adj_close = "close"

            for i in range(1, len(df)):
                if (df[adj_close][i] > df[adj_close][i - 1]):
                    tally += df['volume'][i]
                elif (df[adj_close][i] < df[adj_close][i - 1]):
                    tally -= df['volume'][i]
                new_balance_volume.append(tally)

            df['on_balance_volume'] = new_balance_volume
            minimum = min(df['on_balance_volume'])

            df['on_balance_volume'] = df['on_balance_volume'] - minimum
            df['on_balance_volume'] = (df['on_balance_volume']+1).transform(np.log)
     
        elif indicator == 'macd':
            df['macd'] = stock.get('macd').copy() # from stockstats
            #df['macd'] = TA.MACD(stock)['MACD'].copy() # from finta

        elif indicator == 'macds':
            df['macds'] = stock.get('macds').copy() # from stockstats

        elif indicator == 'macdh':
            df['macdh'] = stock.get('macdh').copy() # from stockstats

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
        
        elif indicator == 'williams_%r':
            df['williams_%r'] = TA.WILLIAMS(stock).copy()

        elif indicator == 'stoch_%k':
            df['stoch_%k'] = TA.STOCH(stock).copy()

        elif indicator == 'stoch_%d':
            df['stoch_%d'] = TA.STOCHD(stock).copy()
           
        elif indicator == 'er':
            df['er'] = TA.ER(stock).copy()
           
        elif indicator == 'stc':
            df['stc'] = TA.STC(stock).copy()
           
        elif indicator == 'atr':
            df['atr'] = TA.ATR(stock).copy()
           
        elif indicator == 'adx':
            df['adx'] = TA.ADX(stock).copy()
           
        elif indicator == 'roc':
            df['roc'] = TA.ROC(stock).copy()

        elif indicator == 'mom':
            df['mom'] = TA.MOM(stock).copy()

        elif indicator == 'simple_rtn':
            df['simple_rtn'] = df['close'].pct_change()

        elif indicator == 'vsa':
            days = [1, 2, 3, 5, 20, 40, 60]
            df = vsa.create_bunch_of_vsa_features(df, days)
            df['outcomes_vsa'] = df.close.pct_change(-1)

        # 'target' is trend_1d with a shift applied in order tp place on the same row the features and the expected trend for the next day
        elif indicator == 'target':
            diff = df["close"] - df["close"].shift(1)
            df["target"] = diff.gt(0).map({False: 0, True: 1}).shift(-1)
            
        else:
            print("!!! add_technical_indicators !!! unknown indicator : {}".format(indicator))
    
    return df

def get_trend_info(df):
    tmp = pd.concat([df['close']], axis=1, keys=['close'])
    tmp = add_technical_indicators(tmp, ["trend_1d"])
    tmp['shift_trend_1d'] = tmp['trend_1d'].shift(-1)
    tmp.dropna(inplace=True)

    tmp['true_positive'] = np.where((tmp['trend_1d'] == 1) & (tmp['shift_trend_1d'] == 1), 1, 0)
    tmp['true_negative'] = np.where((tmp['trend_1d'] == 0) & (tmp['shift_trend_1d'] == 0), 1, 0)
    tmp['false_positive'] = np.where((tmp['trend_1d'] == 1) & (tmp['shift_trend_1d'] == 0), 1, 0)
    tmp['false_negative'] = np.where((tmp['trend_1d'] == 0) & (tmp['shift_trend_1d'] == 1), 1, 0)

    # how many times the trend is up
    trend_counted = tmp['trend_1d'].value_counts(normalize=True)
    trend_ratio = 100 * trend_counted[1]

    # how many times trend today = trend tomorrow
    true_positive = 100*tmp['true_positive'].value_counts(normalize=True)[1]
    true_negative = 100*tmp['true_negative'].value_counts(normalize=True)[1]
    false_positive = 100*tmp['false_positive'].value_counts(normalize=True)[1]
    false_negative = 100*tmp['false_negative'].value_counts(normalize=True)[1]

    return trend_ratio, true_positive, true_negative, false_positive, false_negative

def get_stats_for_trend_up(df, n_forward_days):
    tmp = df.copy()

    indicator = "trend_"+str(n_forward_days)+"d"
    if indicator not in tmp.columns:
        tmp = add_technical_indicators(tmp, [indicator])

    # how many times the trend is up for d+n_forward_days
    trend_counted = tmp[indicator].value_counts(normalize=True)
    trend_ratio = 100 * trend_counted[1]

    return trend_ratio

def get_stats_on_trend_today_equals_trend_tomorrow(df):
    tmp = pd.concat([df['close']], axis=1, keys=['close'])
    tmp = add_technical_indicators(tmp, ["trend_1d"])
    tmp['shift_trend'] = tmp["trend_1d"].shift(-1)
    tmp.dropna(inplace=True)

    tmp['true_positive'] = np.where((tmp["trend_1d"] == 1) & (tmp['shift_trend'] == 1), 1, 0)
    tmp['true_negative'] = np.where((tmp["trend_1d"] == 0) & (tmp['shift_trend'] == 0), 1, 0)
    tmp['false_positive'] = np.where((tmp["trend_1d"] == 1) & (tmp['shift_trend'] == 0), 1, 0)
    tmp['false_negative'] = np.where((tmp["trend_1d"] == 0) & (tmp['shift_trend'] == 1), 1, 0)

    # how many times trend today = trend tomorrow
    true_positive = 100*tmp['true_positive'].value_counts(normalize=True)[1]
    true_negative = 100*tmp['true_negative'].value_counts(normalize=True)[1]
    false_positive = 100*tmp['false_positive'].value_counts(normalize=True)[1]
    false_negative = 100*tmp['false_negative'].value_counts(normalize=True)[1]

    return true_positive, true_negative, false_positive, false_negative
