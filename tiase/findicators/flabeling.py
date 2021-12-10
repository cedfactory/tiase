"""
Data Labeling methode
Reference:    https://towardsdatascience.com/the-triple-barrier-method-251268419dcd
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tiase.fdatapreprocessing import fdataprep

def plot_barriers_out(barriers, filename):
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['font.family'] = 'serif'

    plt.plot(barriers.out, 'bo')

    plt.savefig(filename + '.png')
    plt.clf()

def plot_barriers_dynamic(barriers, t_final, filename):
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

    fig.savefig(filename + '_1.png')

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

    fig.savefig(filename + '_2.png')

# for intraday data
def get_daily_volatility_for_intraday_data(close,span0=100):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    a = df0 -1 #using a variable to avoid the error message.
    df0=pd.Series(close.index[a], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1
    # daily returns
    df0=df0.ewm(span=span0).std()
    return df0

# for daily data
def get_daily_volatility_for_daily_data(close,span0=20):
    # simple percentage returns
    df0=close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0=df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    return df0

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
            # Labeling with decreasing ending
            vert_barrier = daily_volatility.index[len(daily_volatility.index) - 1]
            # Labeling with nan ending
            #vert_barrier = np.nan
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

def get_labels(barriers, label_below=0, label_middle=1, label_above=2):
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
    floating = False
    for i in range(len(barriers.index)):
        start = barriers.index[i]
        end = barriers.vert_barrier[i]
        if pd.notna(end):
            # assign the top and bottom barriers
            top_barrier = barriers.top_barrier[i]
            bottom_barrier = barriers.bottom_barrier[i]
            #set the profit taking and stop loss conditons
            condition_pt = (barriers.price[start: end] >= top_barrier).any()
            condition_sl = (barriers.price[start: end] <= bottom_barrier).any()
            #set the first to reach the barrier
            if condition_pt and condition_sl:
                cpt_date = barriers.index[i]
                condition_pt_loc = False
                j=1
                while (cpt_date <= end) and (condition_pt_loc == False):
                    if(barriers.price[cpt_date] >= top_barrier):
                        condition_pt_loc = cpt_date
                    else:
                        cpt_date = barriers.index[i+j]
                        j=j+1
                cpt_date = barriers.index[i]
                condition_sl_loc = False
                j=1
                while (cpt_date <= end) and (condition_sl_loc == False):
                    if(barriers.price[cpt_date] <= bottom_barrier):
                        condition_sl_loc = cpt_date
                    else:
                        cpt_date = barriers.index[i+j]
                        j=j+1
                if condition_pt_loc < condition_sl_loc:
                    condition_sl = False
                else:
                    condition_pt = False
            #assign the labels
            if condition_pt:
                barriers['out'][i] = label_above
            elif condition_sl:
                barriers['out'][i] = label_below
            else:
                if not floating:
                    barriers['out'][i] = label_middle
                else:
                    price_initial = barriers.price[start]
                    price_final = barriers.price[end]
                    barriers['out'][i] = max([(price_final - price_initial) / (top_barrier - price_initial),
                                            (price_final - price_initial) / (price_initial - bottom_barrier)],
                                            key=abs)

    return barriers

def data_labeling(df, params = None):
    debug = False
    t_final = 10 # how many days we hold the stock which set the vertical barrier
    upper_multiplier = 2
    lower_multiplier = 2
    label_below = 0
    label_middle = 1
    label_above = 2
    if params:
        debug = params.get('labeling_debug', debug)
        t_final = params.get('labeling_t_final', t_final)
        if isinstance(t_final, str):
            t_final = int(t_final)
        upper_multiplier = params.get('labeling_upper_multiplier', upper_multiplier)
        if isinstance(upper_multiplier, str):
            upper_multiplier = float(upper_multiplier)
        lower_multiplier = params.get('labeling_lower_multiplier', lower_multiplier)
        if isinstance(lower_multiplier, str):
            lower_multiplier = float(lower_multiplier)
        label_below = params.get('labeling_label_below', label_below)
        if isinstance(label_below, str):
            label_below = float(label_below)
        label_middle = params.get('labeling_label_middle', label_middle)
        if isinstance(label_middle, str):
            label_middle = float(label_middle)
        label_above = params.get('labeling_label_above', label_above)
        if isinstance(label_above, str):
            label_above = float(label_above)

    price = df["close"].copy()

    #set the boundary of barriers, based on 20 days EWM
    daily_volatility = get_daily_volatility_for_daily_data(price)

    #align the index
    prices = price[daily_volatility.index]

    barriers = get_3_barriers(prices, daily_volatility, t_final, [upper_multiplier, lower_multiplier])

    barriers = get_labels(barriers, label_below, label_middle, label_above)

    if debug:
        plot_barriers_out(barriers, filename="./tmp/labeling_barriers_out")
        plot_barriers_dynamic(barriers, t_final, filename="./tmp/labeling_barriers_dynamic")
        barriers = fdataprep.process_technical_indicators(barriers, ['missing_values']) # shit happens
        barriers.to_csv("./tmp/labeling_barriers.csv")

    df["labeling"] = barriers['out'].copy()
    
    return df
