from importlib import reload
from lib.ml import ml_bilstm
reload(ml_bilstm)
from lib.fimport import fimport
reload(fimport)
from lib.visu import visu
reload(visu)

import pandas as pd
import datetime
import matplotlib.pyplot as plt

import numpy as np


prices = './lib/data/IBM_Prices.csv'
value = "IBM"
df = ml_bilstm.ReadData(prices)

visu.DisplayFromDataframe(df, 'Close')
#visu.DisplayFromDataframe(df, 'Volume')
#df.head()

bilstm = ml_bilstm.BiLSTM()
bilstm.ImportData(df)
bilstm.TrainModel(epochs = 10)
bilstm.DisplayStats()
bilstm.SaveModel('bi_lstm_'+value)

xmlfile = "./bi_lstm_"+value+".xml"
df = ml_bilstm.ReadData(prices)

#importers.DisplayFromDataframe(df, 'Close')
#importers.DisplayFromDataframe(df, 'Volume')
#df.head()

bilstm = ml_bilstm.BiLSTM()
bilstm.ImportData(df)
bilstm.LoadModel(xmlfile)
bilstm.StatsForTrends()
prediction = bilstm.MakeTrendPredictionForNextTick()
print(prediction)
