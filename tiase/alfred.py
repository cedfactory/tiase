import xml.etree.cElementTree as ET
from tiase.fimport import fimport,visu
from tiase.fdatapreprocessing import fdataprep
from tiase.findicators import findicators
from tiase.ml import data_splitter,classifiers_factory,analysis
from datetime import datetime
import math
import os
from rich import print,inspect

def out(msg):
    print(msg)

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
        export_root = ding.get("export", "./")
        if export_root and not os.path.isdir(export_root):
            os.mkdir(export_root)

        def get_full_path(filename):
            return export_root + '/' + filename

        start = datetime.now()
        out(ding_msg)

        # import
        import_node = ding.find('import')
        if import_node is not None:
            value = import_node.get("value", None)
            params = dict()
            for param in ["start", "end", "period"]:
                if param in import_node.attrib:
                    params[param] = import_node.get(param)
            import_filename = import_node.get("filename", None)
            export_filename = import_node.get("export", None)
            if value:
                out("value : {}".format(value))
                df = fimport.get_dataframe_from_yahoo(value, params)
            elif import_filename:
                out("filename : {}".format(import_filename))
                df = fimport.get_dataframe_from_csv(import_filename, params)
                value = "" # value is used as name to export files, so we can't leave it with None value
            out(df.head())

            if export_filename:
                df.to_csv(get_full_path(export_filename))

        # indicators
        features_node = ding.find('features')
        if features_node is not None:
            features = features_node.get("indicators", None)
            target = features_node.get("target", None)
            export_filename = features_node.get("export", None)
            if target:
                all_features = []
                if features:
                    all_features = features.split(',')
                all_features.append(target)

                out("Using the following technical indicators : {}".format(all_features))
                df = findicators.add_technical_indicators(df, all_features)
                findicators.remove_features(df, ["open", "high", "low", "adj_close", "volume", "dividends", "stock_splits"])
                df = fdataprep.process_technical_indicators(df, ['missing_values'])
                if export_filename:
                    df.to_csv(get_full_path(export_filename))
                    for indicator in df.columns:
                        visu.display_from_dataframe(df, indicator, get_full_path(indicator+'.png'))
            out(df.head())

        # preprocessing
        preprocessing_node = ding.find('preprocessing')
        if preprocessing_node:
            export_filename = preprocessing_node.get("export", None)
            
            for preprocess in preprocessing_node:
                # outliers
                if preprocess.tag == "outliers":
                    out(preprocess.get("method", None))
                    method = preprocess.get("method", None)
                    if method:
                        out("[PREPROCESSING] outliers : {}".format(method))
                        df = fdataprep.process_technical_indicators(df, [method])
                        df = fdataprep.process_technical_indicators(df, ['missing_values'])

                # transformation
                elif preprocess.tag == "transformation":
                    method = preprocess.get("method", None)
                    indicators = preprocess.get("indicators", None)
                    if method is not None and indicators is not None:
                        out("[PREPROCESSING] transformation {} for {}".format(method, indicators))
                        indicators = indicators.split(',')
                        df = fdataprep.process_technical_indicators(df, ["transformation_"+method], indicators)
                        df = fdataprep.process_technical_indicators(df, ['missing_values'])

                elif preprocess.tag == "discretization":
                    # discretization
                    indicators = preprocess.get("indicators", None)
                    method = preprocess.get("method", None)
                    if indicators is not None and method is not None:
                        out("[PREPROCESSING] discretization {} for {}".format(method, indicators))
                        indicators = indicators.split(',')
                        df = fdataprep.process_technical_indicators(df, ["discretization_"+method], indicators)

            if export_filename:
                df.to_csv(get_full_path(export_filename))
                for indicator in df.columns:
                    visu.display_from_dataframe(df, indicator, get_full_path(value + '_preprocessing_'+indicator+'.png'))

        # feature engineering
        featureengineering_node = ding.find('featureengineering')
        if featureengineering_node is not None:
            export_filename = featureengineering_node.get("export", None)

            # reduction
            reduction_node = featureengineering_node.find('reduction')
            if reduction_node is not None:
                method = reduction_node.get("method", None)
                if method is not None:
                    out("[FEATURE ENGINEERING] reduction : {}".format(method))

            if export_filename:
                df.to_csv(get_full_path(export_filename))
                for indicator in df.columns:
                        visu.display_from_dataframe(df, indicator, get_full_path(value + '_featureengineering_'+indicator+'.png'))

        # data splitter
        library_data_splitters = {}
        data_splitters_node = ding.find('data_splitters')
        if data_splitters_node:
            for data_splitter_node in data_splitters_node:
                if data_splitter_node.tag != "data_splitter":
                    continue
                data_splitter_id = data_splitter_node.get("id", None)
                data_splitter_type = data_splitter_node.get("type", None)
                data_splitter_seq_len = float(data_splitter_node.get('sequence_length', math.nan))
                if data_splitter_id == None or data_splitter_type == None or math.isnan(data_splitter_seq_len):
                    continue

                if data_splitter_type == "simple":
                    index = float(data_splitter_node.get('index', math.nan))
                    if math.isnan(index) == False:
                        ds = data_splitter.DataSplitterTrainTestSimple(df, target="target", seq_len=21)
                        ds.split(index)
                    else:
                        continue
                elif data_splitter_type == "cross_validation":
                    nb_splits = int(data_splitter_node.get('nb_splits', math.nan))
                    max_train_size = int(data_splitter_node.get('max_train_size', math.nan))
                    test_size = int(data_splitter_node.get('test_size', math.nan))
                    if math.isnan(nb_splits) == False and math.isnan(max_train_size) == False and math.isnan(test_size) == False:
                        ds = data_splitter.DataSplitterForCrossValidation(df.copy(), nb_splits, max_train_size, test_size)
                        ds.split()
                    else:
                        continue
                else:
                    continue

                library_data_splitters[data_splitter_id] = ds
        print(library_data_splitters)

        # learning model
        classifiers_node = ding.find('classifiers')
        if classifiers_node:
            ds = data_splitter.DataSplitterTrainTestSimple(df, target="target", seq_len=21)
            ds.split(0.7)
            library_models = {}
            test_vs_pred = []
            for classifier in classifiers_node:
                classifier_id = classifier.get("id", None)
                out("[CLASSIFIER] Treating {}".format(classifier_id))
                classifier_type = classifier.get("type", None)
                export_filename = classifier.get("export", None)

                parameters_node = classifier.find('parameters')
                params = {}
                if parameters_node:
                    for parameter in parameters_node:
                        parameter_name = parameter.get("name", None)
                        parameter_value = parameter.get("value", None)
                        if parameter_name != None and parameter_value != None:

                            def get_classifier_from_name(classifier_name):
                                classifier_value = library_models[classifier_name]
                                if classifier_value != None:
                                    out("{} found ({})".format(classifier_name, classifier_value))
                                else:
                                    out("!!! {} not found !!!".format(parameter_value))
                                return classifier_value

                            # replace classifier name with classifier model
                            if parameter_name == "classifier":
                                parameter_value = get_classifier_from_name(parameter_value)

                            elif parameter_name == "classifiers":
                                classifier_names = parameter_value.split(',')
                                parameter_value = [(classifier_name, get_classifier_from_name(classifier_name)) for classifier_name in classifier_names]
                                out(parameter_value)

                            if parameter_value:
                                params[parameter_name] = parameter_value

                model = classifiers_factory.ClassifiersFactory.get_classifier(type=classifier_type, params=params)
                model.fit(ds)
                library_models[classifier_id] = model

                model_analysis = model.get_analysis()
                out("Accuracy : {:.2f}".format(model_analysis["accuracy"]))
                out("Precision : {:.2f}".format(model_analysis["precision"]))
                out("Recall : {:.2f}".format(model_analysis["recall"]))
                out("f1_score : {:.2f}".format(model_analysis["f1_score"]))
                if export_filename:
                    model.save(get_full_path(export_filename))

                test_vs_pred.append(analysis.testvspred(classifier_id, model_analysis["y_test"], model_analysis["y_test_prob"]))

            analysis.export_roc_curves(test_vs_pred, export_root + "/roc_curves.png", "")

        end = datetime.now()
        out("elapsed time : {}".format(end-start))

    return 0

def summary():
    classifiers = classifiers_factory.ClassifiersFactory.get_classifiers_list()
    for classifier_name in classifiers:
        print(classifier_name)
    
    return 0