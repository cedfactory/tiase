"""
Data Labeling methode
Reference:    https://towardsdatascience.com/the-triple-barrier-method-251268419dcd
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def setup_plot_display():
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['font.family'] = 'serif'


def plot_barriers_out(barriers, filename="barriers_out"):
    OUT_DIR = "./tmp_data_labeling/"
    if (os.path.isdir(OUT_DIR) == False):
        print("new results directory: ", OUT_DIR)
        os.mkdir(OUT_DIR)

    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['font.family'] = 'serif'

    # fig = plt.figure(figsize=(20, 16))
    # fig = plt.figure(figsize=(16, 9), dpi=300)

    # setup_plot_display()
    plt.plot(barriers.out, 'bo')

    plt.savefig(OUT_DIR + filename + '.png')
    plt.clf()

def plot_barriers_dynamic(barriers, t_final, filename="barriers_dynamic"):
    OUT_DIR = "./tmp_data_labeling/"
    if (os.path.isdir(OUT_DIR) == False):
        print("new results directory: ", OUT_DIR)
        os.mkdir(OUT_DIR)

    fig, ax = plt.subplots()
    ax.set(title='stock price', xlabel='date', ylabel='price')
    ax.plot(barriers.price[100: 200])
    start = barriers.index[120]
    end = barriers.vert_barrier[120]
    upper_barrier = barriers.top_barrier[120]
    lower_barrier = barriers.bottom_barrier[120]
    ax.plot([start, end], [upper_barrier, upper_barrier], 'r--')
    ax.plot([start, end], [lower_barrier, lower_barrier], 'r--')
    ax.plot([start, end], [(lower_barrier + upper_barrier) * 0.5,
                           (lower_barrier + upper_barrier) * 0.5], 'r--')
    ax.plot([start, start], [lower_barrier, upper_barrier], 'r-')
    ax.plot([end, end], [lower_barrier, upper_barrier], 'r-')

    fig.savefig(OUT_DIR + filename + '_1.png')

    # dynamic graph
    fig, ax = plt.subplots()
    ax.set(title='Apple stock price',
           xlabel='date', ylabel='price')
    ax.plot(barriers.price[100: 200])
    start = barriers.index[120]
    end = barriers.index[120 + t_final]
    upper_barrier = barriers.top_barrier[120]
    lower_barrier = barriers.bottom_barrier[120]
    ax.plot(barriers.index[120:120 + t_final + 1], barriers.top_barrier[start:end], 'r--')
    ax.plot(barriers.index[120:120 + t_final + 1], barriers.bottom_barrier[start:end], 'r--')
    ax.plot([start, end], [(lower_barrier + upper_barrier) * 0.5,
                           (lower_barrier + upper_barrier) * 0.5], 'r--')
    ax.plot([start, start], [lower_barrier, upper_barrier], 'r-')
    ax.plot([end, end], [barriers.bottom_barrier[end], barriers.top_barrier[end]], 'r-')

    fig.savefig(OUT_DIR + filename + '_2.png')

def get_daily_vol(close,span0=100):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    a = df0 -1 #using a variable to avoid the error message.
    df0=pd.Series(close.index[a],
                  index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1
    # daily returns
    df0=df0.ewm(span=span0).std()
    return df0

def get_daily_volatility(close,span0=20):
    # simple percentage returns
    df0=close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0=df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    return df0


def get_atr(stock, win=14):
    atr_df = pd.Series(index=stock.index)
    high = pd.Series(stock.high.rolling(win, min_periods=win))
    low = pd.Series(stock.low.rolling(win, min_periods=win))
    close = pd.Series(stock.close.rolling(win, min_periods=win))

    for i in range(len(stock.index)):
        tr = np.max([(high[i] - low[i]), np.abs(high[i] - close[i]), np.abs(low[i] - close[i])], axis=0)
        atr_df[i] = tr.sum() / win

    return atr_df


def get_3_barriers(prices, daily_volatility, t_final, upper_lower_multipliers):
    #create a container
    barriers = pd.DataFrame(columns=['days_passed', 'price', 'vert_barrier', 'top_barrier', 'bottom_barrier'],
                            index = daily_volatility.index)
    for day, vol in daily_volatility.iteritems():
        days_passed = len(daily_volatility.loc[daily_volatility.index[0] : day])
        #set the vertical barrier
        if (days_passed + t_final < len(daily_volatility.index) and t_final != 0):
            vert_barrier = daily_volatility.index[days_passed + t_final]
        else:
            vert_barrier = np.nan
        #set the top barrier
        if upper_lower_multipliers[0] > 0:
            top_barrier = prices.loc[day] + prices.loc[day] * upper_lower_multipliers[0] * vol
        else:
            #set it to NaNs
            top_barrier = pd.Series(index=prices.index)
        #set the bottom barrier
        if upper_lower_multipliers[1] > 0:
            bottom_barrier = prices.loc[day] - prices.loc[day] * upper_lower_multipliers[1] * vol
        else:
            #set it to NaNs
            bottom_barrier = pd.Series(index=prices.index)

        barriers.loc[day, ['days_passed', 'price', 'vert_barrier','top_barrier', 'bottom_barrier']] = days_passed, prices.loc[day], vert_barrier, top_barrier, bottom_barrier

    barriers['out'] = None
    return barriers

def get_labels(barriers):
    '''
    start: first day of the window
    end:last day of the window
    price_initial: first day stock price
    price_final:last day stock price
    top_barrier: profit taking limit
    bottom_barrier:stop loss limt
    condition_pt:top_barrier touching conditon
    condition_sl:bottom_barrier touching conditon
    '''
    for i in range(len(barriers.index)):
        start = barriers.index[i]
        end = barriers.vert_barrier[i]
        if pd.notna(end):
            # assign the initial and final price
            price_initial = barriers.price[start]
            price_final = barriers.price[end]
            # assign the top and bottom barriers
            top_barrier = barriers.top_barrier[i]
            bottom_barrier = barriers.bottom_barrier[i]
            #set the profit taking and stop loss conditons
            condition_pt = (barriers.price[start: end] >= top_barrier).any()
            condition_sl = (barriers.price[start: end] <= bottom_barrier).any()
            #assign the labels
            if condition_pt:
                barriers['out'][i] = 1
            elif condition_sl:
                barriers['out'][i] = -1
            else:
                barriers['out'][i] = max([(price_final - price_initial) / (top_barrier - price_initial),
                                          (price_final - price_initial) / (price_initial - bottom_barrier)],
                                         key=abs)
    return barriers

def data_labeling(df):
    price = df["close"].copy()

    df0 = get_daily_volatility(price)

    df_atr = get_atr(df, 14)

    #set the boundary of barriers, based on 20 days EWM
    daily_volatility = get_daily_volatility(price)
    # how many days we hold the stock which set the vertical barrier
    t_final = 10
    #the up and low boundary multipliers
    upper_lower_multipliers = [2, 2]
    #allign the index
    prices = price[daily_volatility.index]
    # how many days we hold the stock which set the vertical barrier
    t_final = 10

    barriers = get_3_barriers(prices, daily_volatility, t_final, upper_lower_multipliers)
    print(barriers.info())

    barriers = get_labels(barriers)



    plot_barriers_out(barriers, filename="barriers_out")

    plot_barriers_dynamic(barriers, t_final, filename="barriers_dynamic")


