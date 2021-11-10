from . import fprep, fdiscretize


def process_technical_indicators(df, preprocessing, features = []):
    """
    data preprocessing
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """

    # process data indicators
    for preprocess in preprocessing:
        if preprocess == 'missing_values':
            df = fprep.missing_values(df)
        elif preprocess == 'duplicates':
            df = fprep.drop_duplicates(df)
        elif preprocess == 'outliers_normalize_stdcutoff':
            n_sigmas = 2
            df = fprep.normalize_outliers_std_cutoff(df, n_sigmas)
        elif preprocess == 'outliers_cut_stdcutoff':
            n_sigmas = 2
            df = fprep.cut_outliers_std_cutoff(df, n_sigmas)
        elif preprocess == 'outliers_normalize_winsorize':
            outlier_cutoff = 0.03
            df = fprep.normalize_outliers_winsorize(df, outlier_cutoff)
        elif preprocess == 'outliers_normalize_mam':
            # Using Moving Average Mean
            n_sigmas = 2.5
            df = fprep.normalize_outliers_mam(df, n_sigmas)
        elif preprocess == 'outliers_normalize_ema':
            # Using EMA
            n_sigmas = 2.5
            df = fprep.normalize_outliers_ema(df, n_sigmas)
        elif preprocess == 'feature_encoding':
            df = fprep.feature_encoding(df)
        elif preprocess == 'transformation_log':
            df = fprep.data_log_transformation(df, features)
        elif preprocess == 'transformation_x2':
            df = fprep.data_x2_transformation(df, features)
        elif preprocess == 'discretization_supervised':
            df = fdiscretize.data_discretization(df, features)
        elif preprocess == 'discretization_unsupervised':
            strategy = 'uniform'
            nb_bins = 5
            df = fdiscretize.data_discretization_unsupervized(df, features, nb_bins, strategy)

        else:
            print("Warning : preprocessing {} is unknown".format(preprocess))

    return df
