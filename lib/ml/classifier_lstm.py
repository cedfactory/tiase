from ..findicators import *
from ..ml import toolbox,analysis,classifier

from rich import print,inspect

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# Functional API : https://keras.io/guides/functional_api/
from tensorflow import keras
from tensorflow.keras import layers

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

    def create_model(self):
        self.set_train_test_data()
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))

        # create the model
        tf.random.set_seed(20)
        np.random.seed(10)

        self.build_model()

        #print(self.model.summary())
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=10, verbose=0)

        # Final evaluation of the model
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1]*100))
    
    def get_analysis(self):
        self.y_test_prob = self.model.predict(self.X_test)
        self.y_test_pred = (self.y_test_prob > 0.5).astype("int32")
        self.analysis = analysis.classification_analysis(self.model, self.X_test, self.y_test, self.y_test_pred, self.y_test_prob)
        self.analysis["history"] = getattr(self.model, "history", None)
        return self.analysis

class ClassifierLSTM1(ClassifierLSTM):
    def __init__(self, dataframe, params = None):
        super().__init__(dataframe, params)
    
    def build_model(self):
        print("[Build ClassifierLSTM1]")
        # length of the input = seq_len * (#columns in the dataframe - one reserved for the target)
        shapeDim2 = self.seq_len * (self.df.shape[1] - 1)

        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(1, shapeDim2)))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


class ClassifierLSTM2(ClassifierLSTM):
    def __init__(self, dataframe, params = None):
        super().__init__(dataframe, params)
    
    def build_model(self):
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


# Functional API : https://keras.io/guides/functional_api/
class ClassifierLSTM3(ClassifierLSTM):
    def __init__(self, dataframe, params = None):
        super().__init__(dataframe, params)
    
    def build_model(self):
        print("[Build ClassifierLSTM3]")

        # length of the input = seq_len * (#columns in the dataframe - one reserved for the target)
        shapeDim2 = self.seq_len * (self.df.shape[1] - 1)

        inputs = keras.Input(shape=(1, shapeDim2))
        x = layers.LSTM(100, activation="sigmoid")(inputs)
        #x = layers.Dense(64, activation="sigmoid")(x)
        outputs = layers.Dense(1)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

#
# "Predicting the Trend of Stock Market Index Using the Hybrid Neural Network Based on Multiple Time Scale Feature Learning", Yaping Hao - Qiang Gao, 2020
# https://www.mdpi.com/2076-3417/10/11/3961
# https://www.researchgate.net/publication/342045600_Predicting_the_Trend_of_Stock_Market_Index_Using_the_Hybrid_Neural_Network_Based_on_Multiple_Time_Scale_Feature_Learning
#
class ClassifierLSTM_Hao2020(ClassifierLSTM):
    def __init__(self, dataframe, params = None):
        super().__init__(dataframe, params)
    
    def build_model(self):
        print("[Build ClassifierLSTM_Hao2020]")

        # length of the input = seq_len * (#columns in the dataframe - one reserved for the target)
        shapeDim2 = self.seq_len * (self.df.shape[1] - 1)

        inputs = keras.Input(shape=(1, shapeDim2), name="Input")
        convmax2 = layers.Conv1D(96, 3, activation="relu", padding='same')(inputs)
        convmax2 = layers.MaxPooling1D(pool_size=1, padding='same')(convmax2)
        
        convmax3 = layers.Conv1D(96, 3, activation="relu", padding='same')(convmax2)
        convmax3 = layers.MaxPooling1D(pool_size=1, padding='same')(convmax3)
        F3 = layers.LSTM(64, activation="sigmoid")(convmax3)

        F2 = layers.LSTM(64, activation="sigmoid")(convmax2)
        
        F1 = layers.LSTM(64, activation="sigmoid")(inputs)

        concat = layers.concatenate([F1, F2, F3])
 
        outputs = layers.Dense(32)(concat)
        outputs = layers.Dense(16)(outputs)
        outputs = layers.Dense(1, name="Output")(outputs)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

        self.model.save("lstm.hdf5")

# Source : https://towardsdatascience.com/the-beginning-of-a-deep-learning-trading-bot-part1-95-accuracy-is-not-enough-c338abc98fc2
class ClassifierBiLSTM(ClassifierLSTM):
    def __init__(self, dataframe, params = None):
        super().__init__(dataframe, params)
    
    def build_model(self):
        print("[Build ClassifierBiLSTM]")

        # length of the input = seq_len * (#columns in the dataframe - one reserved for the target)
        shapeDim2 = self.seq_len * (self.df.shape[1] - 1)

        in_seq = keras.Input(shape=(1, shapeDim2))
      
        x = layers.Bidirectional(LSTM(self.seq_len, return_sequences=True))(in_seq)
        x = layers.Bidirectional(LSTM(self.seq_len, return_sequences=True))(x)
        x = layers.Bidirectional(LSTM(int(self.seq_len/2), return_sequences=True))(x) 
      
        avg_pool = layers.GlobalAveragePooling1D()(x)
        max_pool = layers.GlobalMaxPooling1D()(x)
        conc = layers.concatenate([avg_pool, max_pool])
        conc = layers.Dense(int(self.seq_len/2), activation="relu")(conc)
        out = layers.Dense(1, activation="sigmoid")(conc)      

        self.model = keras.Model(inputs=in_seq, outputs=out)
        self.model.compile(loss="mse", optimizer="adam", metrics=['mse', 'mae', 'mape'])    

