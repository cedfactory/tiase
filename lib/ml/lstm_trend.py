import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from ml import analysis

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt


#
# ref : https://medium.com/analytics-vidhya/analysis-of-stock-price-predictions-using-lstm-models-f993faa524c4
#
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
        self.X_train, self.y_train, self.X_test, self.y_test, self.test_data = self.train_test_split_preparation(0.7)

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
        self.history = self.model.fit(x=self.X_train, y=self.y_train, batch_size=15, epochs=epochs, shuffle=True, validation_split = 0.1)

    def get_analysis(self):
        self.analysis = analysis.regression_analysis(self.model, self.X_test, self.y_test)
        self.analysis["history"] = self.history
        return self.analysis

    def export_predictions(self, filename):
        y_pred = self.model.predict(self.X_test)
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
        iStart = len(self.df) - (self.seq_len + 1) - 1
        df_temp = self.df[iStart : iStart + self.seq_len + 1].copy()
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

        X_train = np.array([train_arr[i : i + self.seq_len] for i in range(len(train_arr) - self.seq_len)])

        y_train = np.array([train_arr[i + self.seq_len] for i in range(len(train_arr) - self.seq_len)])

        X_test = np.array([test_arr[i : i + self.seq_len] for i in range(len(test_arr) - self.seq_len)])

        y_test = np.array([test_arr[i + self.seq_len] for i in range(len(test_arr) - self.seq_len)])

        return X_train, y_train, X_test, y_test, test_data
