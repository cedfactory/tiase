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
    "FB": "Facebook, Inc.", # https://finance.yahoo.com/quote/FB/?p=FB
    "FAST": "Fastenal Company", # https://finance.yahoo.com/quote/FAST/?p=FAST
    "FISV": "Fiserv, Inc.", # https://finance.yahoo.com/quote/FISV/?p=FISV
    "GILD": "Gilead Sciences, Inc.", # https://finance.yahoo.com/quote/GILD/?p=GILD
    "HAS": "Hasbro, Inc.", # https://finance.yahoo.com/quote/HAS/?p=HAS
    "HSIC": "Henry Schein, Inc.", # https://finance.yahoo.com/quote/HSIC/?p=HSIC
    "HOLX": "Hologic, Inc.", # https://finance.yahoo.com/quote/HOLX/?p=HOLX
    "IDXX": "IDEXX Laboratories, Inc.", # https://finance.yahoo.com/quote/IDXX/?p=IDXX
    "ILMN": "Illumina, Inc.", # https://finance.yahoo.com/quote/ILMN?p=ILMN
    "INCY": "Incyte Corporation", # https://finance.yahoo.com/quote/INCY?p=INCY
    "INTC": "Intel Corporation", # https://finance.yahoo.com/quote/INTC?p=INTC
    "INTU": "Intuit Inc.", # https://finance.yahoo.com/quote/INTU/?p=INTU
    "ISRG": "Intuitive Surgical, Inc.", # https://finance.yahoo.com/quote/ISRG/?p=ISRG
    "JBHT": "J.B. Hunt Transport Services, Inc.", # https://finance.yahoo.com/quote/JBHT/?p=JBHT
    "JD": "JD.com, Inc.", # https://finance.yahoo.com/quote/JD/?p=JD
    #"": "KLA-Tencor Corporation", 
    "KHC": "The Kraft Heinz Company", # https://finance.yahoo.com/quote/KHC/?p=KHC
    "LRCX": "Lam Research Corporation", # https://finance.yahoo.com/quote/LRCX/?p=LRCX
    "LBTYA": "Liberty Global plc", # https://finance.yahoo.com/quote/LBTYA/?p=LBTYA
    "LBTYK": "Liberty Global plc", # https://finance.yahoo.com/quote/LBTYK?p=LBTYK
    "QRTEA": "Qurate Retail, Inc.", # https://finance.yahoo.com/quote/QRTEA?p=QRTEA
    "MAR": "Marriott International, Inc.", # https://finance.yahoo.com/quote/MAR/?p=MAR
    "MXIM": "Maxim Integrated Products, Inc.", # https://finance.yahoo.com/quote/MXIM/?p=MXIM
    "MELI": "MercadoLibre, Inc.", # https://finance.yahoo.com/quote/MELI/?p=MELI
    "MCHP": "Microchip Technology Incorporated", # https://finance.yahoo.com/quote/MCHP/?p=MCHP
    "MU": "Micron Technology, Inc.", # https://finance.yahoo.com/quote/MU/?p=MU
    "MSFT": "Microsoft Corporation", # https://finance.yahoo.com/quote/MSFT/?p=MSFT
    "MDLZ": "Mondelez International, Inc.", # https://finance.yahoo.com/quote/MDLZ/?p=MDLZ
    "MNST": "Monster Beverage Corporation", # https://finance.yahoo.com/quote/MNST/?p=MNST
    #"": "Mylan Laboratories", # 
    "NTES": "NetEase, Inc.", # https://finance.yahoo.com/quote/NTES?p=NTES
    "NFLX": "Netflix, Inc.", # https://finance.yahoo.com/quote/NFLX?p=NFLX
    "NVDA": "NVIDIA Corporation", # https://finance.yahoo.com/quote/NVDA?p=NVDA
    "NXPI": "NXP Semiconductors N.V.", # https://finance.yahoo.com/quote/NXPI/?p=NXPI
    "ORLY": "O'Reilly Automotive, Inc.", # https://finance.yahoo.com/quote/ORLY/?p=ORLY
    "PCAR": "PACCAR Inc", # https://finance.yahoo.com/quote/PCAR/?p=PCAR
    "PAYX": "Paychex, Inc.", # https://finance.yahoo.com/quote/PAYX?p=PAYX
    "PYPL": "PayPal Holdings, Inc.", # https://finance.yahoo.com/quote/PYPL/?p=PYPL
    "PEP": "PepsiCo, Inc.", # https://finance.yahoo.com/quote/PEP/?p=PEP
    "BKNG": "Booking Holdings Inc.", # https://finance.yahoo.com/quote/BKNG/?p=BKNG
    "QCOM": "QUALCOMM Incorporated", # https://finance.yahoo.com/quote/QCOM/?p=QCOM
    "REGN": "Regeneron Pharmaceuticals, Inc.", # https://finance.yahoo.com/quote/REGN/?p=REGN
    "ROST": "Ross Stores, Inc.", # https://finance.yahoo.com/quote/ROST/?p=ROST
    "STX": "Seagate Technology Holdings plc", # https://finance.yahoo.com/quote/STX/?p=STX
    "SIRI": "Sirius XM Holdings Inc.", # https://finance.yahoo.com/quote/SIRI/?p=SIRI
    "SWKS": "Skyworks Solutions, Inc.", # https://finance.yahoo.com/quote/SWKS?p=SWKS
    "SBUX": "Starbucks Corporation", # https://finance.yahoo.com/quote/SBUX/?p=SBUX
    #"": "", # Symantec
    "SNPS": "Synopsys, Inc.", # https://finance.yahoo.com/quote/SNPS?p=SNPS&.tsrc=fin-srch-v1
    "TTWO": "Take-Two Interactive Software, Inc.", # https://finance.yahoo.com/quote/TTWO/?p=TTWO
    "TSLA": "Tesla, Inc.", # https://finance.yahoo.com/quote/TSLA/?p=TSLA
    "TXN": "Texas Instruments Incorporated", # https://finance.yahoo.com/quote/TXN/?p=TXN
    "TMUS": "T-Mobile US, Inc.", # https://finance.yahoo.com/quote/TMUS/?p=TMUS
    "ULTA": "Ulta Beauty, Inc.", # https://finance.yahoo.com/quote/ULTA/?p=ULTA
    "VRSK": "Verisk Analytics, Inc.", # https://finance.yahoo.com/quote/VRSK?p=VRSK
    "VRTX": "Vertex Pharmaceuticals Incorporated", # https://finance.yahoo.com/quote/VRTX/?p=VRTX
    "VOD": "Vodafone Group Plc", # https://finance.yahoo.com/quote/VOD/?p=VOD
    "WBA": "Walgreens Boots Alliance, Inc.", # https://finance.yahoo.com/quote/WBA/?p=WBA
    "WDC": "Western Digital Corporation", # https://finance.yahoo.com/quote/WDC/?p=WDC
    "WDAY": "Workday, Inc.", # https://finance.yahoo.com/quote/WDAY/?p=WDAY
    "WYNN": "Wynn Resorts, Limited", # https://finance.yahoo.com/quote/WYNN/?p=WYNN
    "XLNX": "Xilinx, Inc."  # https://finance.yahoo.com/quote/XLNX/?p=XLNX
}

def DownloadFromYahoo(values, folder = ""):
    for value in values:
        data_df = yf.download(value, period="max")
        filename = './' + folder + '/' + value + '.csv'
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
