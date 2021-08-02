import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_sma_ema_wma(df, columns):
    if "sma" in columns:
        columns = [x for x in columns if x is not "sma"]
        col_sma_lst = [item for item in df.columns if item.startswith("sma")]
        columns.extend(col_sma_lst)

    if "ema" in columns:
        columns = [x for x in columns if x is not "ema"]
        col_ema_lst = [item for item in df.columns if item.startswith("ema")]
        columns.extend(col_ema_lst)

    if "wma" in columns:
        columns = [x for x in columns if x is not "wma"]
        col_wma_lst = [item for item in df.columns if item.startswith("wma")]
        columns.extend(col_wma_lst)

    return columns


def correlation_reduction(df, columns):
    columns = get_sma_ema_wma(df, columns)

    df_copy_target = df['target'].copy()
    df_for_feature_eng = df[columns]
    plt.figure(figsize=(16, 16))
    heatmap = sns.heatmap(df_for_feature_eng.corr(), vmin=-1, vmax=1, annot=True)
    plt.savefig("heatmap_before.png")
    plt.clf()

    data = df[columns]

    # Create correlation matrix
    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # Drop features
    data = data.drop(data[to_drop], axis=1)

    # columns.extend(["target"])

    df_for_feature_eng = df[data.columns]
    plt.figure(figsize=(16, 16))
    heatmap = sns.heatmap(df_for_feature_eng.corr(), vmin=-1, vmax=1, annot=True)
    plt.savefig("heatmap_after.png")
    plt.clf()

    df = df[data.columns]

    frames = [df, df_copy_target]
    df_result = pd.concat(frames, axis=1).reindex(df.index)

    return df_result


def pca_reduction(df, columns):
    columns = get_sma_ema_wma(df, columns)

    print("Feature PCA selection...")
    df_index = df.index.tolist()
    data = df[columns].copy()
    df = df.drop(columns, axis=1)

    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(data)

    pca = PCA(n_components=0.99)
    pca.fit(data_rescaled)
    data_pca = pca.transform(data_rescaled)

    df_pca = pd.DataFrame(data=data_pca, index=df_index)

    pca_col = []
    for col in df_pca.columns:
        pca_col.append("pca_" + str(col))
    df_pca.columns = pca_col

    frames = [df, df_pca]
    df_result = pd.concat(frames, axis=1).reindex(df.index)

    return df_result


def rfecv_reduction(df, columns):
    # model_type = 'XGB'
    model_type = 'SVC'
    scoring = 'accuracy'  # 'precision' , 'f1' , 'recall', 'accuracy'

    # MIN_FEATURE = int(len(columns) * 2 / 3)
    MIN_FEATURE = 3

    print("RFECV Feature selection...")
    print("model: ", model_type)
    print("scoring: ", scoring)

    columns = get_sma_ema_wma(df, columns)

    df_index = df.index.tolist()
    X = df[columns].copy()
    target = df['target']
    df = df.drop(columns, axis=1)

    if (model_type == 'XGB'):
        rfc = XGBClassifier(random_state=101, verbosity=0)
    if (model_type == 'Forest'):
        rfc = RandomForestClassifier(random_state=101)
    if (model_type == 'SVC'):
        rfc = SVC(kernel="linear")

    # rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring=scoring)
    rfecv = RFECV(estimator=rfc, scoring=scoring, min_features_to_select=MIN_FEATURE)
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

    rfecv_col = []
    for col in X.columns:
        rfecv_col.append("rfecv_" + str(col))
    df_rfecv.columns = rfecv_col

    frames = [df, df_rfecv]
    df_result = pd.concat(frames, axis=1).reindex(df.index)

    return df_result
