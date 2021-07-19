import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from findicators import *
from ml import toolbox
from ml import analysis
from ml import classifier

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
class ClassifierLSTM(classifier.Classifier):
    def __init__(self, dataframe, params = None):
        self.seq_len = 21
        self.df = dataframe

        self.df = findicators.add_technical_indicators(self.df, ["trend_1d"])
        self.df = findicators.remove_features(self.df, ["open","close","low","high","volume"])

        self.df.dropna(inplace = True)

        self.epochs = 170
        if "epochs" in params:
            self.epochs = params["epochs"]

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(1, self.seq_len)))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def create_model(self):
        # split the data
        self.X_train, self.y_train, self.X_test, self.y_test, self.x_normaliser = toolbox.get_train_test_data_from_dataframe1(self.df, self.seq_len, 'trend_1d', .7)
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))

        # create the model
        tf.random.set_seed(20)
        np.random.seed(10)

        self.build_model()

        print(self.model.summary())
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=64)

        # Final evaluation of the model
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    def get_analysis(self):
        self.analysis = analysis.classification_analysis(self.model, self.X_test, self.y_test)
        self.analysis["history"] = self.history
        return self.analysis

class ClassifierLSTM2(ClassifierLSTM):
    def __init__(self, dataframe, name = ""):
        super().__init__(dataframe, name)
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(1, self.seq_len)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation='sigmoid'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
