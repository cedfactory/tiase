from lib.fimport import *


df = fimport.GetDataFrameFromYahoo("AI.PA")
visu.DisplayFromDataframe(df,"Close")

y = synthetic.get_sinusoid(length=100, amplitude=1, frequency=.1, phi=0, height = 0)
df = synthetic.create_dataframe(y, .1)
visu.DisplayFromDataframe(df,"Close", "close.png")
