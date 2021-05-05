import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np

#
# https://github.com/yuhaolee97/stock-project/blob/main/basicmodel.py
#
def create_model_lstm(X_train, y_train, seq_len):
    tf.random.set_seed(20)
    np.random.seed(10)
    lstm_input = Input(shape=(seq_len, 6), name='lstm_input')

    inputs = LSTM(21, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr = 0.0008)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=15, epochs=20, shuffle=True, validation_split = 0.1)

    return model
