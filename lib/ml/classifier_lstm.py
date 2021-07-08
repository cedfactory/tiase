import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from findicators import *
from ml import toolbox
from ml import analysis

import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

#
# Simple LSTM for Sequence Classification
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
#
class LSTMClassification:

    def __init__(self, dataframe, name = ""):
        self.seq_len = 21
        self.df = dataframe

        self.df = findicators.add_technical_indicators(self.df, ["trend_1d"])
        self.df = findicators.remove_features(self.df, ["open","close","low","high","volume"])

        self.df.dropna(inplace = True)

    def create_model(self, epochs = 170):

        # split the data
        self.X_train, self.y_train, self.X_test, self.y_test, self.x_normaliser = toolbox.get_train_test_data_from_dataframe1(self.df, self.seq_len, 'trend_1d', .7)
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))

        # create the model
        tf.random.set_seed(20)
        np.random.seed(10)

        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(1, self.seq_len)))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


        '''
        lstm_input = Input(shape=(self.seq_len, 6), name='lstm_input')

        inputs = LSTM(21, name='first_layer')(lstm_input)
        inputs = Dense(15, name='first_dense_layer')(inputs)
        inputs = Dense(1, name='second_dense_layer')(inputs)
        output = Activation('linear', name='output')(inputs)

        self.model = Model(inputs=lstm_input, outputs=output)
        adam = optimizers.Adam(lr = 0.0008)
        self.model.compile(optimizer=adam, loss='mse')
        '''

        '''
        self.model = Sequential()
        self.model.add(Embedding(input_dim = 188, output_dim = 50, input_length = len(self.X_train[0])))
        self.model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        '''
        
        '''
        self.model = Sequential()
        self.model.add(LSTM(units=100, input_shape=(1, self.seq_len)))
        self.model.add(Dense(1, activation='sigmoid'))
        #self.model.add(Dropout(0.5))
        #self.model.add(Dense(100, activation='relu'))
        #self.model.add(Dense(self.X_train.shape[1], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        '''
        print(self.model.summary())
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=epochs, batch_size=64)

        # Final evaluation of the model
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1]*100))


    def get_analysis(self):
        self.analysis = analysis.classification_analysis(self.model, self.X_test, self.y_test)
        self.analysis["history"] = self.history
        return self.analysis
