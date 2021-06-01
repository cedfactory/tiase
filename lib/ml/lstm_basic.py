import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from findicators import *
from ml import toolbox
from ml import analysis

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt

#
# ref : https://medium.com/analytics-vidhya/analysis-of-stock-price-predictions-using-lstm-models-f993faa524c4
#
class LSTMBasic:

    def __init__(self, dataframe, name = ""):
        self.seq_len = 21
        self.df = dataframe

        self.df = findicators.add_technical_indicators(self.df, ["on_balance_volume", "ema","bbands"])
        self.df = findicators.remove_features(self.df, ["open","close","low","high","volume"])


        # fill NaN
        for i in range(19):
            self.df['bb_middle'][i] = self.df['ema'][i]
            
            if i != 0:
                higher = self.df['bb_middle'][i] + 2 * self.df['adj_close'].rolling(i + 1).std()[i]
                lower = self.df['bb_middle'][i] - 2 * self.df['adj_close'].rolling(i + 1).std()[i]
                self.df['bb_upper'][i] = higher
                self.df['bb_lower'][i] = lower
            else:
                self.df['bb_upper'][i] = self.df['bb_middle'][i]
                self.df['bb_lower'][i] = self.df['bb_middle'][i]

        #self.df.dropna(inplace = True)

        if name != "":
            self.load_model(name)

    def create_model(self):

        # build the model
        tf.random.set_seed(20)
        np.random.seed(10)
        lstm_input = Input(shape=(self.seq_len, 6), name='lstm_input')

        inputs = LSTM(21, name='first_layer')(lstm_input)
        inputs = Dense(1, name='dense_layer')(inputs)
        output = Activation('linear', name='output')(inputs)

        self.model = Model(inputs=lstm_input, outputs=output)
        adam = optimizers.Adam(lr = 0.0008)
        self.model.compile(optimizer=adam, loss='mse')

        # training
        self.X_train, self.y_train, self.X_test, self.y_test, self.x_normaliser, self.y_normaliser = toolbox.get_train_test_data_from_dataframe0(self.df, self.seq_len, .7)
        self.model.fit(x=self.X_train, y=self.y_train, batch_size=15, epochs=170, shuffle=True, validation_split = 0.1)

        # analysis
        self.analysis = analysis.regression_analysis(self.model, self.X_test, self.y_test)
        print("mape : {:.2f}".format(self.analysis["mape"]))
        print("rmse : {:.2f}".format(self.analysis["rmse"]))
        print("mse :  {:.2f}".format(self.analysis["mse"]))

        # output
        y_pred = self.model.predict(self.X_test)
        y_pred = self.y_normaliser.inverse_transform(y_pred)
        
        real = plt.plot(self.y_test, label='Actual Price')
        pred = plt.plot(y_pred, label='Predicted Price')

        plt.gcf().set_size_inches(12, 8, forward=True)
        plt.title('Plot of real price and predicted price against number of days')
        plt.xlabel('Number of days')
        plt.ylabel('Adjusted Close Price($)')
        plt.legend(['Actual Price', 'Predicted Price'])
        plt.savefig('lstm.png')



    def save_model(self, name):
            filename = name+'.hdf5'
            self.model.save(filename)

            toolbox.serialize(self.x_normaliser, name+"_x_normalizer.gz")
            toolbox.serialize(self.y_normaliser, name+"_y_normalizer.gz")


            root = ET.Element("model")
            ET.SubElement(root, "file").text = name+'.hdf5'
            ET.SubElement(root, "x_normaliser").text = name+"_x_normalizer.gz"
            ET.SubElement(root, "y_normaliser").text = name+"_y_normalizer.gz"
            ET.SubElement(root, "mape").text = "{:.2f}".format(self.analysis["mape"])
            ET.SubElement(root, "rmse").text = "{:.2f}".format(self.analysis["rmse"])

            xmlfilename = name+'.xml'

            tree = ET.ElementTree(root)

            tree.write(xmlfilename)
            print(xmlfilename)

    def load_model(self, name):
        tree = ET.parse(name+".xml")
        root = tree.getroot()
        
        self.model = tf.keras.models.load_model(root[0].text)
        #self.min_return = float(root[1].text)
        print("Load model from {} : {}".format(name, root[0].text))
        self.x_normaliser = toolbox.deserialize(name+"_x_normalizer.gz")
        self.y_normaliser = toolbox.deserialize(name+"_y_normalizer.gz")


    def predict(self):
        normalised_df = self.x_normaliser.transform(self.df)

        iStart = len(normalised_df) - self.seq_len
        X = np.array([normalised_df[iStart : iStart + self.seq_len].copy()])
        y_normalized = self.model.predict(X)
        y_normalized = np.expand_dims(y_normalized, -1)
        print(y_normalized)
        y = self.y_normaliser.inverse_transform(y_normalized[0])
        print(y)
