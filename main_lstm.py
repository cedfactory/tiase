import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import xml.etree.cElementTree as ET
import tensorflow as tf

# Reference : https://github.com/yuhaolee97/stock-project/blob/main/basicmodel.py

#
# import data
#

df = fimport.GetDataFrameFromCsv("./lib/data/CAC40/AI.PA.csv")

#data = [[10,1000], [2,500], [3,600], [2,800], [2,500], [0,700], [1,800], [3,500], [5,900], [7,1000]]
#df = pd.DataFrame(data, columns = ['indicator', 'Adj Close'])
#df.index.name = 'Date'

#
# add technical indicators
#

def on_balance_volume_creation(df):
    # Adding of on balance volume to dataframe
    new_balance_volume = [0]
    tally = 0

    for i in range(1, len(df)):
        if (df['Adj Close'][i] > df['Adj Close'][i - 1]):
            tally += df['Volume'][i]
        elif (df['Adj Close'][i] < df['Adj Close'][i - 1]):
            tally -= df['Volume'][i]
        new_balance_volume.append(tally)

    df['On_Balance_Volume'] = new_balance_volume
    minimum = min(df['On_Balance_Volume'])

    df['On_Balance_Volume'] = df['On_Balance_Volume'] - minimum
    df['On_Balance_Volume'] = (df['On_Balance_Volume']+1).transform(np.log)

    return df

df = on_balance_volume_creation(df)

df = findicators.add_technical_indicators(df, ["ema","bbands"])
#df = findicators.remove_features(df, ["open","close","low","high","volume"])
df = findicators.remove_features(df, ["open","close","low","high","volume"])
print(df.head(21))

# fill NaN
for i in range(19):
    df['bb_middle'][i] = df['ema'][i]
    
    if i != 0:
        higher = df['bb_middle'][i] + 2 * df['adj_close'].rolling(i + 1).std()[i]
        lower = df['bb_middle'][i] - 2 * df['adj_close'].rolling(i + 1).std()[i]
        df['bb_upper'][i] = higher
        df['bb_lower'][i] = lower
    else:
        df['bb_upper'][i] = df['bb_middle'][i]
        df['bb_lower'][i] = df['bb_middle'][i]

print(df.head(21))

#
# prepare data for validation and testing
#
seq_len = 21
X_train, y_train, X_test, y_test, y_reverse_normaliser = toolbox.get_train_test_data_from_dataframe(df, seq_len, .5)

print(df.head())


#
# create a model
#

model = lstm.lstm_create_model(X_train, y_train, seq_len)

#
# prediction
#

analysis = analysis.regression_analysis(model, X_test, y_test)
print("mape : {:.2f}".format(analysis["mape"]))
print("rmse : {:.2f}".format(analysis["rmse"]))

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

#plt.show()
plt.savefig('lstm.png')

#analysis.make_analysis(model, X_test, y_test)

#
# save the model
#
def SaveModel(model, analysis, name):
    filename = name+'.hdf5'
    model.save(filename)

    root = ET.Element("model")
    ET.SubElement(root, "file").text = name+'.hdf5'
    ET.SubElement(root, "mape").text = "{:.2f}".format(analysis["mape"]) #str(analysis["mape"])
    ET.SubElement(root, "rmse").text = "{:.2f}".format(analysis["rmse"]) #str(analysis["rmse"])

    xmlfilename = name+'.xml'

    tree = ET.ElementTree(root)

    tree.write(xmlfilename)
    print(xmlfilename)

def LoadModel(filename):

    tree = ET.parse(filename)
    root = tree.getroot()
    
    model = tf.keras.models.load_model(root[0].text)
    #self.min_return = float(root[1].text)
    print("Load model from {} : {}".format(filename, root[0].text))
    
    return model
    
SaveModel(model, analysis, "lstm")

model = LoadModel("lstm.xml")
