import xml.etree.cElementTree as ET
from lib.fimport import *
from lib.findicators import *
from lib.ml import *

'''
process :
*** import data ***
- dataframe ohlcv
- simple_rtn & target computation
- [drop ohlvc?]
*** data processing ***
- outliers (\in {stdcutoff, winsorize, mam, ema}
- indicators computation (especially if close are updated by the outliers computation)
- transform \in {log, x2}
- discretization \in {supervised, unsupervised}
*** feature engineering ***
- reduction \in {correlation, pca, rfecv, vsa}
- balancing \in {smote}
*** machine learning ***
- classifier | regressor (\in {svc, xgboost, keras, ...})



input :
- value (AI.PA, ...)
- outliers method
- technical indicators list
- drop ohlcv ?
- transform {"method1" : [indicators], "method2" : [indicators]}
- discretize {"indicator" : method + params (eg strategy for KBinsDiscretizer) + #bins (2 by default)}
- reduction : method + threshold (0.95 by default)
- balancing
- learning model
'''

def execute(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    
    if root.tag != "dings":
        return 1

    for ding in root.findall('ding'):

        for importNode in ding.findall('import'):
            value = importNode.get("value", None)
            if value != None:
                df = fimport.GetDataFrameFromYahoo(value)
                print(value)
                print(df.head())
            break

        for featuresNode in ding.findall('features'):
            features = featuresNode.get("indicators", None)
            target = featuresNode.get("target", None)
            if features != None and target != None:
                features = features.split(',')
                target = target.split(',')
                all = features
                all.extend(target)
                df = findicators.add_technical_indicators(df, all)
                # todo implement findicators.keep([])
                findicators.remove_features(df, ["open", "high", "low", "adj_close", "volume", "dividends", "stock_splits"])
                print(df.head())
            break

        for classifierNode in ding.findall('classifier'):
            classifier_name = classifierNode.get("name", None)
            if classifier_name == 'svc':
                model = classifier_svc.ClassifierSVC(df.copy())
                model.create_model()
                model_analysis = model.get_analysis()
                print("Precision : ", model_analysis["precision"])
                print("Recall : ", model_analysis["recall"])
                print("f1_score:", model_analysis["f1_score"])

            break

    return 0