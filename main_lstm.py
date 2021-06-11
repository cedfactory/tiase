import pandas as pd
from lib.fimport import *
from lib.findicators import *
from lib.ml import *

#
# basic lstm
# Reference : https://github.com/yuhaolee97/stock-project/blob/main/basicmodel.py
#
def basic():
    filename = "./lib/data/CAC40/AI.PA.csv"
    filename = "./google_stocks_data.csv"
    df = fimport.GetDataFrameFromCsv(filename)

    model = lstm_basic.LSTMBasic(df)
    model.create_model()

    analysis = model.get_analysis()
    print("mape : {:.2f}".format(analysis["mape"]))
    print("rmse : {:.2f}".format(analysis["rmse"]))
    print("mse :  {:.2f}".format(analysis["mse"]))

    model.export_predictions("lstm_basic.png")

    model.save_model("basic")

    model_loaded = lstm_basic.LSTMBasic(df, name="basic")
    prediction = model_loaded.predict()
    print(prediction)

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def export_history(history):
    
    if ('accuracy' in history.history.keys()):
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        plt.savefig("_accuracy.png")

    if ('loss' in history.history.keys()):
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        #plt.show()
        plt.savefig("_loss.png")


def trend():
    filename = "./lib/data/test/GOOG.csv"
    filename = "./lib/data/test/google_stocks_data.csv"
    df = fimport.GetDataFrameFromCsv(filename)



    '''
    real = np.array(df['Adj Close'])
    print(real)
    print(real[:-1])
    print(real[1:])

    print(mean_squared_error(real[:-1], real[1:], squared = False))


    real = plt.plot(real[:-1], label='Actual Price')
    pred = plt.plot(real[1:], label='Predicted Price')

    plt.gcf().set_size_inches(12, 8, forward=True)
    plt.title('Plot of real price and predicted price against number of days')
    plt.xlabel('Number of days')
    plt.ylabel('Adjusted Close Price($)')
    plt.legend(['Actual Price', 'Predicted Price'])
    plt.savefig("toto.png")

    return
    '''


    model = lstm_trend.LSTMHaoTrend(df)
    model.create_model(epochs=10)

    analysis = model.get_analysis()
    #print(analysis)
    print("mape : {:.2f}".format(analysis["mape"]))
    print("rmse : {:.2f}".format(analysis["rmse"]))
    print("mse :  {:.2f}".format(analysis["mse"]))

    print(analysis["history"].history.keys())
    export_history(analysis["history"])

    model.export_predictions("lstm_trend.png")

    prediction = model.predict()
    print(prediction)

def classification():
    print("classification")

    filename = "./google_stocks_data.csv"
    df = fimport.GetDataFrameFromCsv(filename)

    model = lstm_classification.LSTMClassification(df)
    model.create_model()


_usage_str = """
Options:
    [ --basic, --trend, --classification]
"""

def _usage():
    print(_usage_str)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--basic": basic()
        elif sys.argv[1] == "--trend": trend()
        elif sys.argv[1] == "--classification": classification()
        else: _usage()
    else: _usage()
