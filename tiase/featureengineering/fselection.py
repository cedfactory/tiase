import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif,f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from rich import print,inspect


def get_sma_ema_wma(df, columns):
    if "sma" in columns:
        columns = [x for x in columns if x != "sma"]
        col_sma_lst = [item for item in df.columns if item.startswith("sma")]
        columns.extend(col_sma_lst)

    if "ema" in columns:
        columns = [x for x in columns if x != "ema"]
        col_ema_lst = [item for item in df.columns if item.startswith("ema")]
        columns.extend(col_ema_lst)

    if "wma" in columns:
        columns = [x for x in columns if x != "wma"]
        col_wma_lst = [item for item in df.columns if item.startswith("wma")]
        columns.extend(col_wma_lst)

    return columns


def kbest_reduction(df, score_func_name, k=0.5, verbose=False):
    if score_func_name == 'f_regression':
        score_func = f_regression
    elif score_func_name == 'f_classif':
        score_func = f_classif
    elif score_func_name == 'chi2':
        score_func = chi2
    else:
        return None

    if k < 1:
        k = int((len(df.columns) - 2) * k)

    df_copy_simple_rtn = df['simple_rtn'].copy()
    x = df.drop(['simple_rtn','target'], axis=1)
    y = df['target'].copy()

    select = SelectKBest(score_func=score_func, k=k)
    fit = select.fit(x, y)

    selected_features = fit.transform(x)
    features_names = np.array(x.columns)
    support = select.get_support()
    selected_columns = features_names[support]
    
    if verbose:
        print("k_best = {}".format(k))
        print("columns : {}".format(features_names))
        print("support : {}".format(support))
        print("score : {}".format(fit.scores_))
        print("selected_columns : {}".format(selected_columns))

    df_x_fitted = pd.DataFrame(index=df.index, data=selected_features, columns=selected_columns)

    df_result = pd.concat([df_copy_simple_rtn, y, df_x_fitted], axis=1)

    return df_result


def correlation_reduction(df):
    list_features = df.columns.to_list()
    list_features.remove('simple_rtn')
    list_features.remove('target')

    df_copy_simple_rtn = df['simple_rtn'].copy()
    df_copy_target = df['target'].copy()

    df_for_feature_eng = df[list_features]

    # Create correlation matrix
    corr_matrix = df_for_feature_eng.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # Drop features
    selected_features = df_for_feature_eng.drop(df_for_feature_eng[to_drop], axis=1)

    frames = [df_copy_simple_rtn, df_copy_target, selected_features]
    df_result = pd.concat(frames, axis=1).reindex(df.index)

    return df_result


def sort_df_by_corr(df, col, method="corr"):
    if method == "corr":
        corr_with = df.corrwith(df[col])
        corr_with = corr_with.sort_values(ascending=False)
        corr_with_col = corr_with.index.tolist()
        results = df[corr_with_col].copy()

        return results


def pca_reduction(df, coef_pca=0.99):
    list_features = df.columns.to_list()
    list_features.remove('simple_rtn')
    list_features.remove('target')

    df_copy_simple_rtn = df['simple_rtn'].copy()
    df_copy_target = df['target'].copy()

    df_for_feature_eng = df[list_features]

    df_index = df.index.tolist()

    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(df_for_feature_eng)

    pca = PCA(n_components=coef_pca)
    pca.fit(data_rescaled)
    data_pca = pca.transform(data_rescaled)

    df_pca = pd.DataFrame(data=data_pca, index=df_index)

    pca_col = []
    for col in df_pca.columns:
        pca_col.append("pca_" + str(col))
    df_pca.columns = pca_col

    frames = [df_copy_simple_rtn, df_copy_target, df_pca]
    df_result = pd.concat(frames, axis=1).reindex(df.index)

    return df_result


def rfecv_reduction(df, model_type, scoring, rfecv_min_features):
    if rfecv_min_features < 1:
        rfecv_min_features = int((len(df.columns) - 2) * rfecv_min_features)

    list_features = df.columns.to_list()
    list_features.remove('simple_rtn')
    list_features.remove('target')

    df_copy_simple_rtn = df['simple_rtn'].copy()
    df_copy_target = df['target'].copy()

    df_for_feature_eng = df[list_features]

    print("RFECV Feature selection...")
    print("model: ", model_type)
    print("scoring: ", scoring)

    X = df_for_feature_eng.copy()
    target = df['target']

    if (model_type == 'XGB'):
        rfc = XGBClassifier(random_state=101, verbosity=0)
    if (model_type == 'Forest'):
        rfc = RandomForestClassifier(random_state=101)
    if (model_type == 'SVC'):
        rfc = SVC(kernel="linear")

    # rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring=scoring)
    rfecv = RFECV(estimator=rfc, scoring=scoring, min_features_to_select=rfecv_min_features)
    rfecv.fit(X, target)

    print('Optimal number of features: {}'.format(rfecv.n_features_))

    plt.figure(figsize=(16, 9))
    plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

    plt.savefig('RFECV_selection.png')
    plt.clf()

    print(np.where(rfecv.support_ == False)[0])

    X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

    df_rfecv = X.copy()

    frames = [df_copy_simple_rtn, df_copy_target, df_rfecv]
    df_result = pd.concat(frames, axis=1).reindex(df.index)

    return df_result


def get_outliers(df, n_sigmas):
    # n_sigmas is number of sigma, which define the boundary along mean

    df['outliers'] = 0

    for col in df.columns:
        mu = df[col].mean()
        sigma = df[col].std()

        # condition = (df[col] > mu + sigma * n_sigmas) | (df[col] < mu - sigma * n_sigmas)
        # outliers[f'{col}_outliers'] = df[col][condition]

        cond = (df[col] > mu + sigma * n_sigmas) | (df[col] < mu - sigma * n_sigmas)
        df['outliers'] = np.where(cond, 1, df['outliers'])

    return df


def vsa_corr_selection(df, debug=False):
    if debug == True:
        folder = './tmp/'
        if (os.path.isdir("vsa_traces") == False):
            print("new vsa traces directory")
            os.mkdir(folder)

    vsa_columns = [feature for feature in df.columns if feature.startswith("vsa_")]
    # check the correlation
    df_vsa = df[vsa_columns]
    corr_vsa = df_vsa.corrwith(df.outcomes_vsa)

    if debug == True:
        plt.figure(figsize=(15, 5))
        corr_vsa.sort_values(ascending=False).plot.barh(title='Strength of Correlation')
        plt.savefig(folder + 'vsa_corr.png')
        plt.clf()

        plt.figure(figsize=(15, 5))

    corr_matrix_vsa = df_vsa.corr()
    corr_matrix_vsa = corr_matrix_vsa.sort_values(ascending=False, by=df_vsa.columns[0])

    if debug == True:
        sns.clustermap(corr_matrix_vsa, cmap='coolwarm', linewidth=1, method='ward')
        plt.savefig(folder + 'vsa_cluster_map.png')
        plt.clf()

    deselected_features_v1 = ['vsa_close_loc_3D', 'vsa_close_loc_60D',
                              'vsa_volume_3D', 'vsa_volume_60D',
                              'vsa_price_spread_3D', 'vsa_price_spread_60D',
                              'vsa_close_change_3D', 'vsa_close_change_60D']

    selected_features_1D_list = ['vsa_volume_2D', 'vsa_price_spread_2D', 'vsa_close_loc_2D', 'vsa_close_change_2D']
    # selected_features_1D_list = ['vsa_volume_1D', 'vsa_price_spread_1D', 'vsa_close_loc_1D', 'vsa_close_change_1D']

    selected_features_1D = df_vsa[selected_features_1D_list]

    selected_features_1D.replace([np.inf, -np.inf], np.nan)
    selected_features_1D.dropna(axis=0, how='any', inplace=True)

    if debug == True:
        sns.pairplot(selected_features_1D)
        plt.savefig(folder + 'vsa_pairplot_1D_map_with_outliers.png')
        plt.clf()

    df = get_outliers(df, 2.5)

    df.drop(df[df['outliers'] == 1].index, inplace=True)

    if debug == True:
        sns.pairplot(df, vars=selected_features_1D_list);
        plt.savefig(folder + 'vsa_pairplot_1D_map_with_no_outliers.png')
        plt.clf()

    df['sign_of_trend'] = df['outcomes_vsa'].apply(np.sign)
    # df['sign_of_trend'] = np.where(df['sign_of_trend'] == 0, 1, df['sign_of_trend'])

    df['sign_of_trend'] = df['sign_of_trend'] * 10

    if debug == True:
        sns.pairplot(df,
                    vars=selected_features_1D_list,
                    diag_kind='kde',
                    # palette='husl',
                    palette='bright',
                    hue='sign_of_trend',
                    # markers=['*', '<', '+'],
                    # markers=['v', '.', 'x'],
                    markers=["o", "s", "D"],
                    plot_kws={'alpha': 0.3})  # transparence:0.3

        plt.savefig(folder + 'vsa_pairplot_2D_final.png')
        plt.clf()

    return df
