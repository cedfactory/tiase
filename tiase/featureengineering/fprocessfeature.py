from . import fselection, fbalance, fstationary, flabeling


def process_features(df, featureengineering, params=None):
    # process data indicators
    for process in featureengineering:
        if process == 'correlation_reduction':
            df = fselection.correlation_reduction(df)
        elif process == 'pca_reduction':
            coef_pca = 0.99
            df = fselection.pca_reduction(df, coef_pca)
        elif process == 'rfecv_reduction':
            model_type = 'Forest'  # 'SVC'
            scoring = 'accuracy'  # 'precision' , 'f1' , 'recall', 'accuracy'
            rfecv_min_features = 3
            df = fselection.rfecv_reduction(df, model_type, scoring, rfecv_min_features)
        elif process == 'smote_balance':
            balance_methode = 'smote'
            df = fbalance.balance_features(df, balance_methode)
        elif process == 'kbest_reduction':
            score_func_name = 'f_classif'
            k = 0.7
            verbose = False
            if params:
                score_func = params.get("score_func_name", score_func)
                k = params.get("k",k)
                verbose = params.get("verbose", verbose)
            df = fselection.kbest_reduction(df, score_func_name=score_func_name, k=k, verbose=verbose)
        elif process == 'vsa_reduction':
            df = fselection.vsa_corr_selection(df)
        elif process == 'stationary_transform':
            df = fstationary.stationary_transform(df)
        elif process == 'data_labeling':
            df = flabeling.data_labeling(df)
        elif process == 'sort_df_by_corr':
            col = 'close'
            method = 'corr'
            df = fselection.sort_df_by_corr(df, col, method)
        else:
            print(":fix: process {} is unknown".format(process))

    return df
