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

# https://www.boursier.com/indices/composition/nasdaq-100-US6311011026,US.html
nasdaq100 = {
    "ATVI": "Activision Blizzard, Inc.", # https://fr.finance.yahoo.com/quote/atvi/?p=atvi
    "ADBE": "Adobe Inc.", # https://fr.finance.yahoo.com/quote/ADBE/?p=ADBE
    "ALXN": "Alexion Pharmaceuticals, Inc.", # https://fr.finance.yahoo.com/quote/alxn/?p=alxn
    "ALGN": "Align Technology, Inc.", # https://fr.finance.yahoo.com/quote/ALGN/?p=ALGN
    "GOOGL": "Alphabet Inc. class A", # https://fr.finance.yahoo.com/quote/googl/?p=googl
    "GOOG": "Alphabet Inc. class C", # https://fr.finance.yahoo.com/quote/goog/?p=goog
    "AMZN": "Amazon.com, Inc.", # https://fr.finance.yahoo.com/quote/AMZN/?p=AMZN
    "AAL" : "American Airlines Group Inc.", # https://fr.finance.yahoo.com/quote/AAL/?p=AAL
    "AMGN": "Amgen Inc.", # https://fr.finance.yahoo.com/quote/AMGN/?p=AMGN
    "ADI": "Analog Devices, Inc.", # https://fr.finance.yahoo.com/quote/ADI/?p=ADI
    "AAPL": "Apple Inc.", # https://fr.finance.yahoo.com/quote/AAPL/?p=AAPL
    "AMAT": "Applied Materials, Inc.", # https://fr.finance.yahoo.com/quote/AMAT/?p=AMAT
    "ASML": "ASML Holding N.V.",  # https://fr.finance.yahoo.com/quote/asml/?p=asml
    "ADSK": "Autodesk, Inc.", # https://fr.finance.yahoo.com/quote/ADSK/?p=ADSK
    "ADP": "Automatic Data Processing, Inc.", # https://fr.finance.yahoo.com/quote/ADP/?p=ADP
    "BIDU": "Baidu, Inc.", # https://fr.finance.yahoo.com/quote/BIDU/history?p=BIDU
    "BIIB": "Biogen Inc.", # https://fr.finance.yahoo.com/quote/BIIB/history?p=BIIB
    "BMRN": "BioMarin Pharmaceutical Inc.", # https://fr.finance.yahoo.com/quote/BMRN/history?p=BMRN
    "CDNS": "Cadence Design Systems, Inc.", # https://fr.finance.yahoo.com/quote/CDNS/history?p=CDNS
    #"": "", # Celgene
    "CERN": "Cerner Corporation", # https://finance.yahoo.com/quote/CERN?p=CERN
    "CHTR": "Charter Communications, Inc.", # https://finance.yahoo.com/quote/CHTR/?p=CHTR
    "CHKP": "Check Point Software Technologies Ltd.", # https://finance.yahoo.com/quote/CHKP/?p=CHKP
    "CTAS": "Cintas Corporation", # https://finance.yahoo.com/quote/CTAS/?p=CTAS
    "CSCO": "Cisco Systems, Inc.", # https://finance.yahoo.com/quote/CSCO/?p=CSCO
    "CTXS": "Citrix Systems, Inc.", # https://finance.yahoo.com/quote/CTXS/?p=CTXS
    "CTSH": "Cognizant Technology Solutions Corporation", # https://finance.yahoo.com/quote/CTSH?p=CTSH
    "CMCSA": "Comcast Corporation", # https://finance.yahoo.com/quote/CMCSA/?p=CMCSA
    "COST": "Costco Wholesale Corporation", # https://finance.yahoo.com/quote/COST/?p=COST
    "CSX": "CSX Corporation", # https://finance.yahoo.com/quote/CSX/?p=CSX
    #"": "", # CTRIP.COM INTERNAT ADR 1/4 SH
    "XRAY": "DENTSPLY SIRONA Inc.", # https://finance.yahoo.com/quote/XRAY?p=XRAY
    "DLTR": "Dollar Tree, Inc.", # https://finance.yahoo.com/quote/DLTR/?p=DLTR
    "EBAY": "eBay Inc.", # https://finance.yahoo.com/quote/EBAY/?p=EBAY
    "EA": "Electronic Arts Inc.", # https://finance.yahoo.com/quote/EA/?p=EA
    "EXPE": "Expedia Group, Inc.", # https://finance.yahoo.com/quote/EXPE?p=EXPE
    #"": "" # Express Scripts
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
    dataframe.dropna(inplace=True) # remove incoherent values (null, NaN, ...)
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
    elif values == "nasdaq100":
        DownloadFromYahoo(nasdaq100.keys())

    
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test1" or sys.argv[1] == "-t1": _test1()
        elif sys.argv[1] == "--test2" or sys.argv[1] == "-t2": _test2()
        elif sys.argv[1] == "--cac40" : _download('cac40')
        elif sys.argv[1] == "--nasdaq100" : _download('nasdaq100')
        else: _usage()
    else: _usage()
