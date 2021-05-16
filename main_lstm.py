import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *

# Reference : https://github.com/yuhaolee97/stock-project/blob/main/basicmodel.py

#
# import data
#

df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/_AI.PA.csv")

#data = [[10,1000], [2,500], [3,600], [2,800], [2,500], [0,700], [1,800], [3,500], [5,900], [7,1000]]
#df = pd.DataFrame(data, columns = ['indicator', 'Adj Close'])
#df.index.name = 'Date'

def on_balance_volume_creation(stock_df):
    # Adding of on balance volume to dataframe
    
    new_df = pd.DataFrame({})

    new_df = stock_df[['Adj Close']].copy()


    new_balance_volume = [0]
    tally = 0

    for i in range(1, len(new_df)):
        if (stock_df['Adj Close'][i] > stock_df['Adj Close'][i - 1]):
            tally += stock_df['Volume'][i]
        elif (stock_df['Adj Close'][i] < stock_df['Adj Close'][i - 1]):
            tally -= stock_df['Volume'][i]
        new_balance_volume.append(tally)

    new_df['On_Balance_Volume'] = new_balance_volume
    minimum = min(new_df['On_Balance_Volume'])

    new_df['On_Balance_Volume'] = new_df['On_Balance_Volume'] - minimum
    new_df['On_Balance_Volume'] = (new_df['On_Balance_Volume']+1).transform(np.log)

    return new_df

#
# add technical indicators
#

df = findicators.add_technical_indicators(df, ["ema","bbands"])
df = findicators.remove_features(df, ["open","close","low","high","volume"])
print(df.head(21))

# fill NaN
for i in range(19):
    df['bb_middle'][i] = df['ema'][i]
    
    if i != 0:
        higher = df['bb_middle'][i] + 2 * df['adj close'].rolling(i + 1).std()[i]
        lower = df['bb_middle'][i] - 2 * df['adj close'].rolling(i + 1).std()[i]
        df['bb_upper'][i] = higher
        df['bb_lower'][i] = lower
    else:
        df['bb_upper'][i] = df['bb_middle'][i]
        df['bb_lower'][i] = df['bb_middle'][i]

print(df.head(21))

#
# prepare data for validation and testing
#
seq_len = 20
X_train, y_train, X_test, y_test, y_normaliser = toolbox.get_train_test_data_from_dataframe(df, seq_len, .5)

print(df.head())


#
# create a model
#

model = lstm.lstm_create_model(X_train, y_train, seq_len)

#
# prediction
#

y_pred = model.predict(X_test)
y_pred = y_reverse_normaliser.inverse_transform(y_pred)

real = plt.plot(y_test, label='Actual Price')
pred = plt.plot(y_pred, label='Predicted Price')

plt.gcf().set_size_inches(12, 8, forward=True)
plt.title('Plot of real price and predicted price against number of days')
plt.xlabel('Number of days')
plt.ylabel('Adjusted Close Price($)')

plt.legend(['Actual Price', 'Predicted Price'])

print(mean_squared_error(y_test, y_pred))

plt.show()    
