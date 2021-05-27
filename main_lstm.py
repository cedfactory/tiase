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

df = findicators.add_technical_indicators(df, ["on_balance_volume", "ema","bbands"])
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
X_train, y_train, X_test, y_test, x_normaliser, y_normaliser = toolbox.get_train_test_data_from_dataframe(df, seq_len, .5)

print(df.head())


#
# create a model
#

model = lstm.lstm_create_model(X_train, y_train, seq_len)

#
# prediction
#

result = analysis.regression_analysis(model, X_test, y_test)
print("mape : {:.2f}".format(result["mape"]))
print("rmse : {:.2f}".format(result["rmse"]))

y_pred = model.predict(X_test)
y_pred = y_normaliser.inverse_transform(y_pred)

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

#
# save the model
#
def SaveModel(model, result, name):
    filename = name+'.hdf5'
    model.save(filename)

    root = ET.Element("model")
    ET.SubElement(root, "file").text = name+'.hdf5'
    ET.SubElement(root, "mape").text = "{:.2f}".format(result["mape"]) #str(result["mape"])
    ET.SubElement(root, "rmse").text = "{:.2f}".format(result["rmse"]) #str(result["rmse"])

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
    


SaveModel(model, result, "lstm")
toolbox.serialize(x_normaliser, "lstm_x_normalizer.gz")
toolbox.serialize(y_normaliser, "lstm_y_normalizer.gz")

######
print("deserialization")
model2 = LoadModel("lstm.xml")

x_normaliser2 = toolbox.deserialize("lstm_x_normalizer.gz")
y_normaliser2 = toolbox.deserialize("lstm_y_normalizer.gz")

result = analysis.regression_analysis(model2, X_test, y_test)
print("mape : {:.2f}".format(result["mape"]))
print("rmse : {:.2f}".format(result["rmse"]))



train_normalised_data = x_normaliser2.transform(df)

iStart = len(train_normalised_data) - seq_len
X = np.array([train_normalised_data[iStart : iStart + seq_len].copy()])
y_normalized = model2.predict(X)
y_normalized = np.expand_dims(y_normalized, -1)
print(y_normalized)
y = y_normaliser2.inverse_transform(y_normalized[0])
print(y)
