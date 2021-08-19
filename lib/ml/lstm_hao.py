from ..findicators import findicators
from . import toolbox,analysis

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
class LSTMHaoBasic:

    def __init__(self, dataframe, name = ""):
        self.seq_len = 21
        self.df = dataframe

        self.df = findicators.add_technical_indicators(self.df, ["on_balance_volume", "ema_9","bbands"])
        self.df = findicators.remove_features(self.df, ["open","close","low","high","volume"])


        # fill NaN
        for i in range(19):
            self.df['bb_middle'][i] = self.df['ema_9'][i]
            
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

    def create_model(self, epochs = 170):

        # split the data
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_normaliser, self.y_normaliser = toolbox.get_train_test_data_from_dataframe0(self.df, self.seq_len, 'adj_close', .7)

        # build the model
        tf.random.set_seed(20)
        np.random.seed(10)
        lstm_input = Input(shape=(self.seq_len, 6), name='lstm_input')

        inputs = LSTM(21, name='first_layer')(lstm_input)
        inputs = Dense(15, name='first_dense_layer')(inputs)
        inputs = Dense(1, name='second_dense_layer')(inputs)
        output = Activation('linear', name='output')(inputs)

        self.model = Model(inputs=lstm_input, outputs=output)
        adam = optimizers.Adam(lr = 0.0008)
        self.model.compile(optimizer=adam, loss='mse')

        # training
        self.history = self.model.fit(x=self.x_train, y=self.y_train, batch_size=15, epochs=epochs, shuffle=True, validation_split = 0.1)

    def get_analysis(self):
        self.analysis = analysis.regression_analysis(self.model, self.x_test, self.y_test, self.y_normaliser)
        self.analysis["history"] = self.history
        return self.analysis


    def export_predictions(self, filename):
        y_pred = self.model.predict(self.x_test)
        y_pred = self.y_normaliser.inverse_transform(y_pred)
        
        real = plt.plot(self.y_test, label='Actual Price')
        pred = plt.plot(y_pred, label='Predicted Price')

        plt.gcf().set_size_inches(12, 8, forward=True)
        plt.title('Plot of real price and predicted price against number of days')
        plt.xlabel('Number of days')
        plt.ylabel('Adjusted Close Price($)')
        plt.legend(['Actual Price', 'Predicted Price'])
        plt.savefig(filename)



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
        print("Load model from {} : {}".format(name, root[0].text))
        self.x_normaliser = toolbox.deserialize(name+"_x_normalizer.gz")
        self.y_normaliser = toolbox.deserialize(name+"_y_normalizer.gz")


    def predict(self):
        normalised_df = self.x_normaliser.transform(self.df)

        i_start = len(normalised_df) - self.seq_len
        X = np.array([normalised_df[i_start : i_start + self.seq_len].copy()])
        y_normalized = self.model.predict(X)
        y_normalized = np.expand_dims(y_normalized, -1)
        y = self.y_normaliser.inverse_transform(y_normalized[0])
        return y[0][0]


class LSTMHaoTrend:

    def __init__(self, dataframe, name = ""):
        self.seq_len = 21
        self.df = dataframe

        # prepare data
        self.df = self.df[['Adj Close']].copy()


    def create_model(self, epochs = 25):
        # Setting of seed (to maintain constant result)
        tf.random.set_seed(20)
        np.random.seed(10)

        # Train test split
        self.x_train, self.y_train, self.x_test, self.y_test, self.test_data = self.train_test_split_preparation(0.7)

        # Build the LSTM model
        lstm_input = Input(shape=(self.seq_len, 1), name='input_for_lstm')

        inputs = LSTM(21, name='first_layer', return_sequences = True)(lstm_input)

        inputs = Dropout(0.1, name='first_dropout_layer')(inputs)
        inputs = LSTM(32, name='lstm_1')(inputs)
        inputs = Dropout(0.05, name='lstm_dropout_1')(inputs) # Dropout layers to prevent overfitting
        inputs = Dense(32, name='first_dense_layer')(inputs)
        inputs = Dense(1, name='dense_layer')(inputs)
        output = Activation('linear', name='output')(inputs)

        self.model = Model(inputs=lstm_input, outputs=output)
        adam = optimizers.Adam(lr = 0.002)

        self.model.compile(optimizer=adam, loss='mse')
        self.history = self.model.fit(x=self.x_train, y=self.y_train, batch_size=15, epochs=epochs, shuffle=True, validation_split = 0.1)

    def get_analysis(self):
        self.analysis = analysis.regression_analysis(self.model, self.x_test, self.y_test)
        self.analysis["history"] = self.history
        return self.analysis

    def export_predictions(self, filename):
        y_pred = self.model.predict(self.x_test)
        y_pred = y_pred.flatten()
        
        # actual represents the test set's actual stock prices
        actual = np.array([self.test_data['Adj Close'][i + self.seq_len] for i in range(len(self.test_data) - self.seq_len)])

        # Adding each actual price at time t with the predicted difference to get a predicted price at time t + 1
        temp_actual = actual[:-1]
        new = np.add(temp_actual, y_pred)

        real = plt.plot(actual[1:], label='Actual Price')
        pred = plt.plot(new, label='Predicted Price')

        plt.gcf().set_size_inches(12, 8, forward=True)
        plt.title('Plot of real price and predicted price against number of days')
        plt.xlabel('Number of days')
        plt.ylabel('Adjusted Close Price($)')
        plt.legend(['Actual Price', 'Predicted Price'])
        plt.savefig(filename)

    def predict(self):
        i_start = len(self.df) - (self.seq_len + 1) - 1
        df_temp = self.df[i_start : i_start + self.seq_len + 1].copy()
        X = np.array([np.diff(df_temp.loc[:, ['Adj Close']].values, axis = 0)])
        y = self.model.predict(X)
        return df_temp['Adj Close'].iloc[-1] + y[0][0]

    def train_test_split_preparation(self, train_split):
        self.df = self.df[1:]

        # Preparation of train test set.
        train_indices = int(self.df.shape[0] * train_split)
        train_data = self.df[:train_indices]
        test_data = self.df[train_indices:]
        test_data = test_data.reset_index()
        test_data = test_data.drop(columns = ['Date'])

        train_arr = np.diff(train_data.loc[:, ['Adj Close']].values, axis = 0)
        test_arr = np.diff(test_data.loc[:, ['Adj Close']].values, axis = 0)

        x_train = np.array([train_arr[i : i + self.seq_len] for i in range(len(train_arr) - self.seq_len)])

        y_train = np.array([train_arr[i + self.seq_len] for i in range(len(train_arr) - self.seq_len)])

        x_test = np.array([test_arr[i : i + self.seq_len] for i in range(len(test_arr) - self.seq_len)])

        y_test = np.array([test_arr[i + self.seq_len] for i in range(len(test_arr) - self.seq_len)])

        return x_train, y_train, x_test, y_test, test_data
