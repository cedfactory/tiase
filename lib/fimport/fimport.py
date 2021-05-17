import yfinance as yf
import pandas as pd
import datetime

cac40 = {
    "AI.PA": "Air Liquide",
    "AIR.PA": "Airbus Group",
    "ALO.PA": "ALSTOM",
    "MT": "ArcelorMittal",
    "ATOS": "Atos",
    "CS.PA": "AXA",
    "BNP.PA": "BNP Paribas",
    "EN.PA": "Bouygues",
    "CAP.PA": "Capgemini",
    "CA.PA": "Carrefour",
    "ACA.PA": "Credit Agricole",
    "BN.PA": "Danone",
    "DSY.PA": "Dassault Systemes",
    "ENGI.PA": "Engie",
    "EL.PA": "EssilorLuxottica",
    "RMS.PA": "Hermes International",
    "KER.PA": "Kering",
    "LR.PA": "Legrand",
    "OR.PA": "L'Oréal",
    "MC.PA": "LVMH",
    "ML.PA": "Michelin",
    "ORA.PA": "Orange",
    "RI.PA": "Pernod Ricard",
    "PUB.PA": "Publicis",
    "RNO.PA": "Renault",
    "SAF.PA": "Safran",
    "SGO.PA": "Saint-Gobain",
    "SAN.PA": "Sanofi",
    "SU.PA": "Schneider Electric",
    "GLE.PA": "Société Générale",
    "STLA": "Stellantis",
    "STM.PA": "STMicroElectronics",
    "TEP.PA": "Teleperformance",
    "HO.PA": "Thales",
    "FP.PA": "Total",
    "URW.AS": "Unibail-Rodamco-Westfield",
    "VIE.PA": "Veolia Environnement",
    "DG.PA": "Vinci",
    "VIV.PA": "Vivendi",
    "WLN.PA": "Worldline"
}

def DownloadFromYahoo(values):
    for value in values:
        data_df = yf.download(value, period="max")
        filename = './' + value + '.csv'
        print(filename)
        data_df.to_csv(filename)

def GetDataFrameFromYahoo(value):
    result = yf.Ticker(value)
    #print(result.info)
    hist = result.history(period="max")
    return hist

def GetDataFrameFromCsv(csvfile):
    dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
    dataframe = pd.read_csv(csvfile,parse_dates=[0],index_col=0,skiprows=0,date_parser=dateparse)
    #df.index.rename('Time',inplace=True)
    #openValues2 = df.sort_values(by='Time')['open'].to_numpy()
    dataframe = dataframe.dropna() # remove incoherent values (null, ...)
    return dataframe


###
### AS A SCRIPT
### python -m import.py
###

_usage_str = """
Options:
    [ --test, -t]
"""

def _usage():
    print(_usage_str)

def _test1():
    hist = GetDataFrameFromYahoo('AI.PA')
    print(hist)

def _test2():
    hist = GetDataFrameFromCsv('./data/AI.PA.csv')
    DisplayFromDataframe(hist, "Close")

def _download(values):
    if values == "cac40":
        DownloadFromYahoo(cac40.keys())

    
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test1" or sys.argv[1] == "-t1": _test1()
        elif sys.argv[1] == "--test2" or sys.argv[1] == "-t2": _test2()
        elif sys.argv[1] == "--cac40" : _download('cac40')
        else: _usage()
    else: _usage()
