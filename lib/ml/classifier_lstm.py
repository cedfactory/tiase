import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from findicators import *
from ml import toolbox,analysis,classifier

import numpy as np
import tensorflow as tf
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
        self.df = dataframe

        self.epochs = 170
        self.seq_len = 21
        if params:
            self.epochs = params.get("epochs", self.epochs)
            self.seq_len = params.get("seq_len", self.seq_len)

    def __build_model(self):
        print("[Build ClassifierLSTM1]")
        # length of the input = seq_len * (#columns in the dataframe - one reserved for the target)
        shapeDim2 = self.seq_len * (self.df.shape[1] - 1)

        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(1, shapeDim2)))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def create_model(self):
        self.set_train_test_data()
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))

        # create the model
        tf.random.set_seed(20)
        np.random.seed(10)

        self.__build_model()

        print(self.model.summary())
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=10, verbose=0)

        # Final evaluation of the model
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1]*100))

class ClassifierLSTM2(ClassifierLSTM):
    def __init__(self, dataframe, params = None):
        super().__init__(dataframe, params)
    
    def __build_model(self):
        print("[Build ClassifierLSTM2]")
        # length of the input = seq_len * (#columns in the dataframe - one reserved for the target)
        shapeDim2 = self.seq_len * (self.df.shape[1] - 1)

        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(1, shapeDim2)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation='sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
