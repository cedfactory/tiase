import pandas as pd

def AddTrend(df, columnSource, columnTarget):
    diff = df[columnSource] - df[columnSource].shift(1)
    df[columnTarget] = diff.gt(0).map({False: 0, True: 1})
    return df
    
