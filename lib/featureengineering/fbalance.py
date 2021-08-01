import pandas as pd

from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE


def smote_balance(df):
    # input DataFrame
    # X → Independent Variable in DataFrame\
    # y → Dependent Variable in Pandas DataFrame format
    BALANCE_METHODE = True
    target = 'target'
    data = df.copy()

    X = data.drop(target, axis=1)
    X_columns = X.columns
    y = data[target].copy()

    X = X.to_numpy()
    y = y.to_numpy()

    if (BALANCE_METHODE == True):
        oversampling = SMOTE()
    else:
        oversampling = ADASYN()

    X, y = oversampling.fit_resample(X, y)

    X_df = pd.DataFrame(X, columns=X_columns)
    y_df = pd.DataFrame(y, columns=[target])

    frame = [X_df, y_df]
    df = pd.concat(frame, axis=1)

    return df
