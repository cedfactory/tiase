"""
    Fractional Differentiation
    https://medium.com/swlh/fractionally-differentiated-features-9c1947ed2b55
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def plotWeights(w, v, filename):
    OUT_DIR = "./tmp_plotweights/"
    if (os.path.isdir(OUT_DIR) == False):
        print("new results directory: ", OUT_DIR)
        os.mkdir(OUT_DIR)

    fig = plt.figure(figsize=(20, 16))
    plt.plot(w)
    plt.xlabel('Number of data points', fontsize=18)
    plt.ylabel('Weight', fontsize=16)

    plt.legend(w.columns, loc='upper right')
    fig.savefig(OUT_DIR + filename + '.png')

    plt.clf()

def plotWeights(out, filename):
    OUT_DIR = "./tmp_plotweights/"
    if (os.path.isdir(OUT_DIR) == False):
        print("new results directory: ", OUT_DIR)
        os.mkdir(OUT_DIR)

    fig = plt.figure(figsize=(20, 16))

    ax1 = out['corr'].plot(figsize=(16, 9), color='r')
    ax2 = out['adfStat'].plot(secondary_y=True, fontsize=20, color='g', ax=ax1)
    ax1.set_title('AdfStat and Corr', fontsize=28)
    ax1.set_xlabel('d value', fontsize=24)
    ax1.set_ylabel('corr', fontsize=24)
    ax2.set_ylabel('adfStat', fontsize=24)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='lower left', fontsize=20)
    plt.axhline(out['95% conf'].mean(),linewidth=3, color='m',linestyle='--');

    fig.savefig(OUT_DIR + filename + '.png')

    plt.clf()

def getWeights(d, size):
    '''
    d:fraction
    k:the number of samples
    w:weight assigned to each samples

    '''
    # thres>0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        tmp = -w[-1]
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)  # sort and reshape the w
    return w


def weight_by_d(dRange=[0, 1], nPlots=11, size=15):
    '''
    dRange: the range of d
    nPlots: the number of d we want to check
    size: the data points used as an example
    w: collection of w by different d value
    '''

    w = pd.DataFrame()

    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0]) \
            [::-1], columns=[d])
        w = w.join(w_, how='outer')

    return w


def get_skip(series, d=0.1, thres=.01):
    '''
    This part is independent of stock price data.
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    '''

    # 1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]

    return skip


def transfor_data_by_frac_diff(col, d=0.1, thres=.01):
    # 3) Apply weights to values
    df = pd.Series()
    skip = get_skip(col)

    for i in range(skip, col.shape[0]):
        i_index = col.index[i]
        data = np.dot(w[-(i + 1):, :].T, col.loc[:i_index])[0]

        df[i_index] = data

    return df


def trans_a_bunch_of_data(df, d=0.1, thres=.01):
    a_bunch_of_trans_data = pd.DataFrame()

    for col in df.columns:
        trans_data = transfor_data_by_frac_diff(df[col], d=d, thres=thres)
        a_bunch_of_trans_data[col] = trans_data

    return a_bunch_of_trans_data


def getWeights_FFD(d=0.1, thres=1e-5):
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres: break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def transfer_data_by_frac_diff_FFD(col, d=0.1, thres=1e-4):
    # 3) Apply weights to values
    w = getWeights_FFD(d, thres)
    width = len(w) - 1

    df = pd.Series()
    # widow size can't be larger than the size of data
    if width >= col.shape[0]: raise Exception("width is oversize")

    for i in range(width, col.shape[0]):
        i_0_index, i_1_index = col.index[i - width], col.index[i]
        data = np.dot(w.T, col.loc[i_0_index:i_1_index])[0]

        df[i_1_index] = data

    return df


def trans_a_bunch_of_data_FFD(df, d=0.1, thres=1e-4):
    a_bunch_of_trans_data = pd.DataFrame()

    for col in df.columns:
        trans_data = transfer_data_by_frac_diff_FFD(df[col], d=d, thres=thres)
        a_bunch_of_trans_data[col] = trans_data

    return a_bunch_of_trans_data


def get_adf_corr(price):
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])

    price_log = np.log(price)
    # price_log = np.log(price[['Close']]).resample('1D').last()  # downcast to daily obs

    for d in np.linspace(0, 1, 11):
        price_trans = transfer_data_by_frac_diff_FFD(price_log, d=d, thres=1e-4)
        # price_trans = transfer_data_by_frac_diff_FFD(price_log, d=d, thres=.01)
        corr = np.corrcoef(price.loc[price_trans.index], price_trans)[0, 1]
        adf = adfuller(price_trans, maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(adf[:4]) + [adf[4]['5%']] + [corr]

    # with critical value
    return out

def select_d_FFD(df, select_best_corr=True):
    df_index = df.index
    index_rounded = []

    for i in df_index:
        index_rounded.append(round(i,1))
    df.index = index_rounded

    # pVal <= 0.05: Reject the null hypothesis(H0), the data does not have a unit root and is stationary
    df = df.drop(df[df['pVal'] > 0.05].index)
    df = df.drop(df[df['corr'] < 0.80].index)

    # df = df.sort_values(['corr', 'pVal'], ascending=[False, True])
    if (select_best_corr == True):
        df = df.sort_values(['corr'], ascending=[False])
    else:
        df = df.sort_values(['pVal'], ascending=[True])

    return df.index[0]


def stationary_transform(df):
    columns = ['open', 'close', 'high', 'low', 'volume']
    df_stock = df[columns].copy()
    price = df['close'].copy()

    out = get_adf_corr(price)
    plotWeights(out, "FFD_d")

    d = select_d_FFD(out)

    df_FFD_transformed = trans_a_bunch_of_data_FFD(df_stock, d=0.1, thres=1e-4)


    w_FFD = getWeights_FFD(thres=1e-4)
    print('FF shape:', w_FFD.shape)

    price_trans = transfer_data_by_frac_diff_FFD(price)
    print('FFD price shape: ', price.shape)
    print('FFD price shape: ', price_trans.shape)

    """
    w = getWeights(0.1, price.shape[0])

    weight_by_d_0_1 = weight_by_d([0, 1])
    plotWeights(weight_by_d_0_1, v=2, filename='weight_by_d_0_1')

    weight_by_d_1_2 = weight_by_d([1, 2])
    plotWeights(weight_by_d_1_2, v=3, filename='weight_by_d_1_2')

    weight_by_d_2_0 = weight_by_d([-2, 0])
    plotWeights(weight_by_d_2_0, v=3, filename='weight_by_d_-2_0')

    weight_by_d_2_3 = weight_by_d([2, 3])
    plotWeights(weight_by_d_2_3, v=3, filename='weight_by_d_2_3')

    weight_by_d_3_4 = weight_by_d([3, 4])
    plotWeights(weight_by_d_3_4, v=3, filename='weight_by_d_3_4')

    weight_by_d_2_4 = weight_by_d([2, 4])
    plotWeights(weight_by_d_2_4, v=3, filename='weight_by_d_2_4')

    weight_by_d_1_5 = weight_by_d([1, 5])
    plotWeights(weight_by_d_1_5, v=3, filename='weight_by_d_1_5')

    weight_by_d_0_10 = weight_by_d([0, 10])
    plotWeights(weight_by_d_0_10, v=3, filename='weight_by_d_0_10')

    weight_by_d_6_7 = weight_by_d([6, 7], nPlots=1)
    plotWeights(weight_by_d_6_7, v=3, filename='weight_by_d_6_7')

    weight_by_d_10_11 = weight_by_d([10, 11], nPlots=1, size=12)
    plotWeights(weight_by_d_10_11, v=10, filename='weight_by_d_10_11')
    """

