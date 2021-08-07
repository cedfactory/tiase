import xml.etree.cElementTree as ET
from lib.fimport import *
from lib.fdatapreprocessing import *
from lib.featureengineering import *
from lib.findicators import *
from lib.ml import *

from rich import print,inspect

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

    ding_msg = '''
   _ _         
 _| |_|___ ___ 
| . | |   | . |
|___|_|_|_|_  |
          |___|'''
    for ding in root.findall('ding'):
        print(ding_msg)

        # import
        importNode = ding.find('import')
        if importNode is not None:
            value = importNode.get("value", None)
            if value != None:
                df = fimport.GetDataFrameFromYahoo(value)
                print(value)
                print(df.head())

        # indicators
        featuresNode = ding.find('features')
        if featuresNode is not None:
            features = featuresNode.get("indicators", None)
            target = featuresNode.get("target", None)
            if features != None and target != None:
                features = features.split(',')
                target = target.split(',')
                all = features
                all.extend(target)
                print("Using the following technical indicators : {}".format(all))
                df = findicators.add_technical_indicators(df, all)
                df = fdataprep.process_technical_indicators(df, ['missing_values'])
                # todo implement findicators.keep([])
                findicators.remove_features(df, ["open", "high", "low", "adj_close", "volume", "dividends", "stock_splits"])
                print(df.head())

        # preprocessing
        preprocessingNode = ding.find('preprocessing')
        if preprocessingNode is not None:
            # outliers
            outliersNode = preprocessingNode.find('outliers')
            if outliersNode is not None:
                print(outliersNode.get("method", None))
                method = outliersNode.get("method", None)
                if method is not None:
                    print("[PREPROCESSING] outliers : {}".format(method))
                    df = fdataprep.process_technical_indicators(df, [method])
                    print(df.head())

            # transformations
            transformationsNode = preprocessingNode.find('transformations')
            if transformationsNode is not None:
                transformations = transformationsNode.findall('transformation')
                for transformation in transformations:
                    method = transformation.get("method", None)
                    indicators = transformation.get("indicators", None)
                    if method is not None and indicators is not None:
                        print("[PREPROCESSING] transformation {} for {}".format(method, indicators))

            # discretizations
            discretizationsNode = preprocessingNode.find('discretizations')
            if discretizationsNode is not None:
                discretizations = discretizationsNode.findall('discretization')
                for discretization in discretizations:
                    indicator = discretization.get("indicator", None)
                    method = transformation.get("method", None)
                    if indicator is not None and method is not None:
                        print("[PREPROCESSING] discretization {} for {}".format(indicator, method))

        # feature engineering
        featureengeineeringNode = ding.find('featureengeineering')
        if featureengeineeringNode is not None:
            # reduction
            reductionNode = featureengeineeringNode.find('reduction')
            if reductionNode is not None:
                method = reductionNode.get("method", None)
                if method is not None:
                    print("[FEATURE ENGINEERING] reduction : {}".format(method))

        # learning model
        classifierNode = ding.find('classifier')
        if classifierNode is not None:
            classifier_name = classifierNode.get("name", None)
            if classifier_name == 'svc':
                model = classifier_svc.ClassifierSVC(df.copy())
                model.create_model()
                model_analysis = model.get_analysis()
                print("Precision : {:.2f}".format(model_analysis["precision"]))
                print("Recall : {:.2f}".format(model_analysis["recall"]))
                print("f1_score : {:.2f}".format(model_analysis["f1_score"]))

    return 0
