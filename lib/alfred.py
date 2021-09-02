import xml.etree.cElementTree as ET
from lib.fimport import fimport,visu
from lib.fdatapreprocessing import fdataprep
from lib.findicators import findicators
from lib.ml import classifier_svc

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
        import_node = ding.find('import')
        if import_node is not None:
            value = import_node.get("value", None)
            if value != None:
                df = fimport.get_dataframe_from_yahoo(value)
                print(value)
                print(df.head())

        # indicators
        features_node = ding.find('features')
        if features_node is not None:
            features = features_node.get("indicators", None)
            target = features_node.get("target", None)
            if features != None and target != None:
                features = features.split(',')
                target = target.split(',')
                all_features = features
                all_features.extend(target)
                target = target[0] # keep the only one target
                print("Using the following technical indicators : {}".format(all_features))
                df = findicators.add_technical_indicators(df, all_features)
                df = fdataprep.process_technical_indicators(df, ['missing_values'])
                # todo implement findicators.keep([])
                findicators.remove_features(df, ["open", "high", "low", "adj_close", "volume", "dividends", "stock_splits"])
                print(df.head())

        # preprocessing
        preprocessing_node = ding.find('preprocessing')
        if preprocessing_node is not None:
            # outliers
            outliers_node = preprocessing_node.find('outliers')
            if outliers_node is not None:
                print(outliers_node.get("method", None))
                method = outliers_node.get("method", None)
                if method is not None:
                    print("[PREPROCESSING] outliers : {}".format(method))
                    df_original = df.copy()
                    df = fdataprep.process_technical_indicators(df, [method])
                    visu.display_outliers_from_dataframe(df_original, df, './tmp/' + value + '_'+method+'.png')
                    df = fdataprep.process_technical_indicators(df, ['missing_values'])
                    print(df.head())

            # transformations
            transformations_node = preprocessing_node.find('transformations')
            if transformations_node is not None:
                transformations = transformations_node.findall('transformation')
                for transformation in transformations:
                    method = transformation.get("method", None)
                    indicators = transformation.get("indicators", None)
                    if method is not None and indicators is not None:
                        print("[PREPROCESSING] transformation {} for {}".format(method, indicators))

            # discretizations
            discretizations_node = preprocessing_node.find('discretizations')
            if discretizations_node is not None:
                discretizations = discretizations_node.findall('discretization')
                for discretization in discretizations:
                    indicator = discretization.get("indicator", None)
                    method = transformation.get("method", None)
                    if indicator is not None and method is not None:
                        print("[PREPROCESSING] discretization {} for {}".format(indicator, method))

        # feature engineering
        featureengeineering_node = ding.find('featureengeineering')
        if featureengeineering_node is not None:
            # reduction
            reduction_node = featureengeineering_node.find('reduction')
            if reduction_node is not None:
                method = reduction_node.get("method", None)
                if method is not None:
                    print("[FEATURE ENGINEERING] reduction : {}".format(method))

        # learning model
        classifier_node = ding.find('classifier')
        if classifier_node is not None:
            classifier_name = classifier_node.get("name", None)
            if classifier_name == 'svc':
                model = classifier_svc.ClassifierSVC(df.copy(), target=target)
                model.create_model()
                model_analysis = model.get_analysis()
                print("Precision : {:.2f}".format(model_analysis["precision"]))
                print("Recall : {:.2f}".format(model_analysis["recall"]))
                print("f1_score : {:.2f}".format(model_analysis["f1_score"]))

    return 0
