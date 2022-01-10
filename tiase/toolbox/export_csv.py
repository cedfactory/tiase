

def make_report(values_classifiers_results, filename):
    f = open(filename, "w")
    f.write("value;classifier id;Accuracy;Precision;Recall;F1 score\n")

    for value in values_classifiers_results:
        classifiers_results = values_classifiers_results[value]
        for classifier_id in classifiers_results:
            result = classifiers_results[classifier_id]
            f.write("{};{};{};{};{};{}\n".format(value, classifier_id, result["accuracy"], result["precision"], result["recall"], result["f1_score"]))

    f.close()
