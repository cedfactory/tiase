import xml.etree.cElementTree as ET
from lib.fimport import *
from lib.fdatapreprocessing import *
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

    for ding in root.findall('ding'):

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
                print(all)
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

            # transformations
            transformationsNode = preprocessingNode.find('transformations')
            if transformationsNode is not None:
                transformations = transformationsNode.findall('transformation')
                for transformation in transformations:
                    print("transformation {} for {}".format(transformation.get("method", None), transformation.get("indicators", None)))

            # discretizations
            discretizationsNode = preprocessingNode.find('discretizations')
            if discretizationsNode is not None:
                discretizations = discretizationsNode.findall('discretization')
                for discretization in discretizations:
                    print("discretization {} for {}".format(discretization.get("indicator", None), transformation.get("method", None)))

        # feature engineering
        featureengeineeringNode = ding.find('featureengeineering')
        if featureengeineeringNode is not None:
            # reduction
            reductionNode = featureengeineeringNode.find('reduction')
            if reductionNode is not None:
                print("reduction : {}".format(reductionNode.get("method", None)))

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
