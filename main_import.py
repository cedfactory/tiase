from lib.fimport import *
from lib.visu import *

df = fimport.GetDataFrameFromYahoo("AI.PA")
visu.DisplayFromDataframe(df,"Close")
