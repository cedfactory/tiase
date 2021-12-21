import xml.etree.cElementTree as ET
from tiase.fimport import fimport,visu
from tiase.fdatapreprocessing import fdataprep
from tiase.featureengineering import fprocessfeature
from tiase.findicators import findicators
from tiase.ml import data_splitter,classifiers_factory,analysis
from datetime import datetime
import pandas as pd
import math
import os
from rich import print,inspect

step_format = "bold red"
def out(msg, format=None):
    if format:
        print("["+format+"]"+ msg + "[/"+format+"]")
    else:
        if isinstance(msg, pd.DataFrame):
            with pd.option_context('display.max_rows',10):
                print(msg)
        else:
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
        # global variables for the current ding execution
        debug = ding.get("debug", False)
        target = None

        export_root = ding.get("export", "./")
        if export_root and not os.path.isdir(export_root):
            os.mkdir(export_root)

        def get_full_path(filename):
            return export_root + '/' + filename

        start = datetime.now()
        out(ding_msg)

        #
        # import
        #
        out("\U0001F449 [IMPORT]", step_format)
        dict_values = dict()
        import_node = ding.find('import')
        if import_node is not None:
            attributes_values = import_node.get("value", None)
            values = None
            if attributes_values:
                values = attributes_values.split(',')
            params = dict()
            for param in ["start", "end", "period"]:
                if param in import_node.attrib:
                    params[param] = import_node.get(param)
            import_filename = import_node.get("filename", None)
            export_filename = import_node.get("export", None)

            if values:
                if "cac40" in values:
                    values.remove("cac40")
                    for key in fimport.cac40.keys():
                        if key not in values:
                            values.append(key)

                for value in values:
                    out("value : {}".format(value))
                    df = fimport.get_dataframe_from_yahoo(value, params)

                    if not isinstance(df, pd.DataFrame):
                        out("\U0001F4A5 no input data for {}".format(value))
                        continue

                    if export_filename:
                        if len(values) > 1:
                            export_filename = value + '_' + export_filename
                        df.to_csv(get_full_path(export_filename))

                    df = findicators.normalize_column_headings(df)
                    dict_values[value] = df

            elif import_filename:
                out("filename : {}".format(import_filename))
                df = fimport.get_dataframe_from_csv(import_filename, params)
                if not isinstance(df, pd.DataFrame):
                    out("\U0001F4A5 no input data for {}".format(value))
                    continue
                
                if export_filename:
                        df.to_csv(get_full_path(export_filename))

                df = findicators.normalize_column_headings(df)

                value = "" # value is used as name to export files, so we can't leave it with None value
                dict_values[value] = df
            
        out("dict_values : {}".format(dict_values.keys()))

        values_classifiers_results = {}
        for key in dict_values:
            current_value = key
            df = dict_values[key]
            out("\U0001F7E2 dealing with {}".format(current_value))
            if df.empty:
                out("\U0001F4A5 Empty DataFrame")
                continue
            out(df)

            values_classifiers_results[current_value] = {}

            initial_columns = list(df.columns)

            # indicators
            out("\U0001F449 [INDICATORS]", step_format)
            features_node = ding.find('features')
            if features_node is not None:
                params = dict()
                for name, value in features_node.attrib.items():
                    if name != "indicators" and name != "target" and name != "export":
                        params[name] = value
                        print("   {} : {}".format(name, value))
                features = features_node.get("indicators", None)
                target = features_node.get("target", None)
                export_filename = features_node.get("export", None)

                all_features = []
                if features:
                    all_features = features.split(',')
                if target != None:
                    all_features.append(target)

                out("Using the following technical indicators : {}".format(all_features))
                df = findicators.add_technical_indicators(df, all_features, params)
                features_to_remove = [feature for feature in initial_columns if feature not in all_features]
                findicators.remove_features(df, features_to_remove)
                df = fdataprep.process_technical_indicators(df, ['missing_values'])
                if export_filename:
                    df.to_csv(get_full_path(export_filename))
                    for indicator in df.columns:
                        visu.display_from_dataframe(df, indicator, get_full_path(indicator+'.png'))
                out(df)

            # preprocessing
            out("\U0001F449 [PREPROCESSING]", step_format)
            preprocessing_node = ding.find('preprocessing')
            if preprocessing_node:
                export_filename = preprocessing_node.get("export", None)
                
                for preprocess in preprocessing_node:
                    # outliers
                    if preprocess.tag == "outliers":
                        method = preprocess.get("method", None)
                        indicators = preprocess.get("indicators", None)
                        if indicators:
                            indicators = indicators.split(',')
                        if method:
                            out("[PREPROCESSING] outliers : {}".format(method))
                            df = fdataprep.process_technical_indicators(df, [method], indicators)
                            df = fdataprep.process_technical_indicators(df, ['missing_values'])

                    # transformation
                    elif preprocess.tag == "transformation":
                        method = preprocess.get("method", None)
                        indicators = preprocess.get("indicators", None)
                        if indicators:
                            indicators = indicators.split(',')
                        if method is not None and indicators is not None:
                            out("[PREPROCESSING] transformation {} for {}".format(method, indicators))
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
            out("\U0001F449 [FEATURE ENGINEERING]", step_format)
            featureengineering_node = ding.find('featureengineering')
            if featureengineering_node is not None:
                export_filename = featureengineering_node.get("export", None)

                for featureengineering in featureengineering_node:
                    # reduction
                    if featureengineering.tag == "reduction":
                        method = featureengineering.get("method", None)
                        if method is not None:
                            out("[FEATURE ENGINEERING] reduction : {}".format(method))

                if export_filename:
                    df.to_csv(get_full_path(export_filename))
                    for indicator in df.columns:
                            visu.display_from_dataframe(df, indicator, get_full_path(value + '_featureengineering_'+indicator+'.png'))

            out("\U0001F449 [FINAL DATAFRAME]", step_format)
            if df.empty:
                out("\U0001F4A5 Empty DataFrame")
                continue
            out(df)

            # target
            out("\U0001F449 [TARGET]", step_format)
            if target == None:
                target_node = ding.find('target')
                if target_node is not None:
                    target = target_node.text
                    shift = target_node.get("shift", None)
                    if shift:
                        df = findicators.shift(df, target, shift)
                        df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens
                        
                    export_filename = target_node.get("export", None)
                    if export_filename:
                        df.to_csv(get_full_path(export_filename))

            if target == None:
                out("\U0001F4A5 no target")
                continue
            out("target : {}".format(target))
            out(df[target].value_counts())

            # in the following, the target column should be named "target"
            df = df.rename(columns={target: "target"})
            target="target"

            # data splitter
            library_data_splitters = {}
            data_splitters_node = ding.find('data_splitters')
            if data_splitters_node:
                for data_splitter_node in data_splitters_node:
                    if data_splitter_node.tag != "data_splitter":
                        continue
                    data_splitter_id = data_splitter_node.get("id", None)
                    data_splitter_type = data_splitter_node.get("type", None)
                    data_splitter_seq_len = int(data_splitter_node.get('sequence_length', math.nan))
                    if data_splitter_id == None or data_splitter_type == None or math.isnan(data_splitter_seq_len):
                        continue

                    if data_splitter_type == "simple":
                        index = float(data_splitter_node.get('index', math.nan))
                        if math.isnan(index) == False:
                            ds = data_splitter.DataSplitterTrainTestSimple(df, target=target, seq_len=data_splitter_seq_len)
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

            # dump data splitters library
            out("\U0001F449 [DATA SPLITTERS LIBRARY]", step_format)
            #print(library_data_splitters)
            for data_splitter_id, data_splitter_item in library_data_splitters.items():
                if isinstance(data_splitter_item, data_splitter.DataSplitterTrainTestSimple):
                    print("DataSplitterTrainTestSimple {}".format(data_splitter_id))
                    #data_splitter_item.dump()
                if isinstance(data_splitter_item, data_splitter.DataSplitterForCrossValidation):
                    print("DataSplitterForCrossValidation {}".format(data_splitter_id))
    
            # learning model
            out("\U0001F449 [CLASSIFIERS]", step_format)
            classifiers_node = ding.find('classifiers')
            if classifiers_node:
                library_models = {}
                test_vs_pred = []
                for classifier in classifiers_node:
                    classifier_id = classifier.get("id", None)
                    out("[CLASSIFIER] Treating {}".format(classifier_id), "red")
                    classifier_type = classifier.get("type", None)
                    data_splitter_id = classifier.get("data_splitter_id", None)
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
                    
                    model = None
                    current_data_splitter = library_data_splitters[data_splitter_id]
                    if current_data_splitter:
                        model = classifiers_factory.ClassifiersFactory.get_classifier(type=classifier_type, params=params)
                        if model == None:
                            out("!!! model {} ({}) is None".format(classifier_id, classifier_type))
                            continue
                        if isinstance(current_data_splitter, data_splitter.DataSplitterTrainTestSimple):
                            model.fit(current_data_splitter)
                        elif isinstance(current_data_splitter, data_splitter.DataSplitterForCrossValidation):
                            model.evaluate_cross_validation(current_data_splitter, target, debug)
                        library_models[classifier_id] = model
                    else:
                        out("!!! can't find data splitter {}".format(data_splitter_id))

                    model_analysis = model.get_analysis()

                    out("Accuracy : {:.2f}".format(model_analysis["accuracy"]))
                    out("Precision : {:.2f}".format(model_analysis["precision"]))
                    out("Recall : {:.2f}".format(model_analysis["recall"]))
                    out("f1_score : {:.2f}".format(model_analysis["f1_score"]))
                    values_classifiers_results[current_value][classifier_id] = model_analysis

                    if export_filename:
                        model.save(get_full_path(export_filename))

                    test_vs_pred.append(analysis.testvspred(classifier_id, model_analysis["y_test"], model_analysis["y_test_prob"]))

                analysis.export_roc_curves(test_vs_pred, export_root + "/roc_curves_"+current_value+".png", "")
            
            analysis.export_classifiers_performances(values_classifiers_results, export_root + "/values_classifiers_results.csv")

        end = datetime.now()
        out("\U0001F3C1 elapsed time : {}".format(end-start), step_format)

    return 0

def summary():
    classifiers = classifiers_factory.ClassifiersFactory.get_classifiers_list()
    return classifiers

def details_for_value(value, root='./tmp/'):
    name = ""
    df = None
    if value.endswith(".csv"):
        name = value
        df = fimport.get_dataframe_from_csv(value)
        basename = os.path.basename(value)
        value = os.path.splitext(basename)[0]
    else:
        if value in fimport.cac40:
            name = fimport.cac40[value]
        elif value in fimport.nasdaq100:
            name = fimport.nasdaq100[value]
        df = fimport.get_dataframe_from_yahoo(value)
    
    if df.empty:
        return False

    print("{} ({})".format(value, name))

    technical_indicators = findicators.get_all_default_technical_indicators()
    df = findicators.add_technical_indicators(df, technical_indicators)
    df = fdataprep.process_technical_indicators(df, ['missing_values']) # shit happens

    trend_ratio_1d = findicators.get_stats_for_trend_up(df, 1)
    trend_ratio_7d = findicators.get_stats_for_trend_up(df, 7)
    trend_ratio_21d = findicators.get_stats_for_trend_up(df, 21)
    print("{} ({});{:.2f};{:.2f};{:.2f}".format(value, name, trend_ratio_1d, trend_ratio_7d, trend_ratio_21d))

    true_positive, true_negative, false_positive, false_negative = findicators.get_stats_on_trend_today_equals_trend_tomorrow(df)
    print("{:.2f},{:.2f},{:.2f},{:.2f}".format(true_positive, true_negative, false_positive, false_negative))

    # format for images
    prefix = value + '_'

    # simple_rtn & histogram
    simple_rtn = df["simple_rtn"].to_numpy()

    visu.display_histogram_fitted_gaussian(simple_rtn, export_name = root + prefix + "simple_rtn_histogram_gaussian.png")
    visu.display_histogram_from_dataframe(df, "simple_rtn", export_name = root + prefix + "simple_rtn_histogram.png")

    # output index.html
    f = open(root + "/index_"+value+".html", "w")
    f.write('<!DOCTYPE html><html lang="en">')
    f.write("<head><title>"+value+"</title></head>")
    f.write("<body>")
    f.write("<h1>"+value+" ("+name+")</h1>")

    f.write('<h3>trends</h3>')
    f.write("<p>trend ratio d+1 : {:.2f}%</p>".format(trend_ratio_1d))
    f.write("<p>trend ratio d+7 : {:.2f}%</p>".format(trend_ratio_7d))
    f.write("<p>trend ratio d+21 : {:.2f}%</p>".format(trend_ratio_21d))

    f.write('<h3>simple_rtn</h3>')
    f.write('<p>mean : '+str(round(simple_rtn.mean(), 6))+'</p>')
    f.write('<p>ratio # positive trends / # trends : '+str(round(len(simple_rtn[simple_rtn > 0])/len(simple_rtn), 6))+'</p>')

    f.write('<p>mean of positive trend : '+str(round(simple_rtn[simple_rtn > 0].mean(), 6))+'</p>')
    f.write('<p>mean of negative trend : '+str(round(simple_rtn[simple_rtn < 0].mean(), 6))+'</p>')
    f.write('<p>histogram :<br><img alt="simple_rtn_histogram_gaussian" width=25% src=' + prefix + "simple_rtn_histogram_gaussian.png" + ' />')
    f.write('<br>')

    f.write('Indicators : <br>')
    for column in df.columns:
        imgname = column + '.png'
        visu.display_from_dataframe(df, column, root + prefix + imgname)
        f.write('<img alt="'+column+'" width=50% src=' + prefix + imgname + ' />')

    f.write("</body></html>")
    f.close()

    return True
