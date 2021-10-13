import xml.etree.cElementTree as ET
from tiase.fimport import fimport,visu
from tiase.fdatapreprocessing import fdataprep
from tiase.findicators import findicators
from tiase.ml import classifier_svc,classifier_lstm

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
            import_filename = import_node.get("filename", None)
            if value != None:
                print("value : {}".format(value))
                df = fimport.get_dataframe_from_yahoo(value)
            elif import_filename != None:
                print("filename : {}".format(import_filename))
                df = fimport.get_dataframe_from_csv(import_filename)
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
                # todo implement findicators.keep([])
                findicators.remove_features(df, ["open", "high", "low", "adj_close", "volume", "dividends", "stock_splits"])
                df = fdataprep.process_technical_indicators(df, ['missing_values'])

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
                    df = fdataprep.process_technical_indicators(df, [method])
                    #visu.display_outliers_from_dataframe(df_original, df, './tmp/' + value + '_'+method+'.png')
                    df = fdataprep.process_technical_indicators(df, ['missing_values'])

            # transformations
            transformations_node = preprocessing_node.find('transformations')
            if transformations_node is not None:
                transformations = transformations_node.findall('transformation')
                for transformation in transformations:
                    method = transformation.get("method", None)
                    indicators = transformation.get("indicators", None)
                    if method is not None and indicators is not None:
                        print("[PREPROCESSING] transformation {} for {}".format(method, indicators))
                        indicators = indicators.split(',')
                        df = fdataprep.process_technical_indicators(df, ["transformation_"+method], indicators)
                        df = fdataprep.process_technical_indicators(df, ['missing_values'])

            # discretizations
            discretizations_node = preprocessing_node.find('discretizations')
            if discretizations_node is not None:
                discretizations = discretizations_node.findall('discretization')
                for discretization in discretizations:
                    indicators = discretization.get("indicators", None)
                    method = discretization.get("method", None)
                    if indicators is not None and method is not None:
                        print("[PREPROCESSING] discretization {} for {}".format(method, indicators))
                        indicators = indicators.split(',')
                        df = fdataprep.process_technical_indicators(df, ["discretization_"+method], indicators)
                        df = fdataprep.process_technical_indicators(df, ['missing_values'])

        # feature engineering
        featureengeineering_node = ding.find('featureengeineering')
        if featureengeineering_node is not None:
            # reduction
            reduction_node = featureengeineering_node.find('reduction')
            if reduction_node is not None:
                method = reduction_node.get("method", None)
                if method is not None:
                    print("[FEATURE ENGINEERING] reduction : {}".format(method))

        # export
        export_node = ding.find('export')
        if export_node is not None:
            export_filename = export_node.get("filename", None)
            df.to_csv(export_filename)

        # learning model
        classifiers_node = ding.find('classifiers')
        if classifiers_node:
            classifiers = classifiers_node.findall('classifier')
            for classifier in classifiers:
                classifier_name = classifier.get("name", None)
                export_filename = classifier.get("export", None)
                print(classifier_name)
                if classifier_name == 'svc':
                    model = classifier_svc.ClassifierSVC(df.copy(), target=target)
                elif classifier_name == "lstm1":
                    model = classifier_lstm.ClassifierLSTM1(df.copy(), target, params={'epochs': 20})
                model.fit()
                model_analysis = model.get_analysis()
                print("Precision : {:.2f}".format(model_analysis["precision"]))
                print("Recall : {:.2f}".format(model_analysis["recall"]))
                print("f1_score : {:.2f}".format(model_analysis["f1_score"]))
                if export_filename:
                    print("export => {}".format(export_filename))
                    model.save(export_filename)

    return 0
