from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph,Table,TableStyle,Image
from reportlab.lib.styles import getSampleStyleSheet

def float_to_string(value, n_decimals=4):
    return '{:.{n_decimals}f}'.format(value, n_decimals=n_decimals)

def make_report(values_classifiers_results, filename):

    # format data to be treated by the reportlab Table
    data = []
    data.append(["value", "classifier id", "Accuracy", "Precision", "Recall", "F1 score"])
    for value in values_classifiers_results:
        classifiers_results = values_classifiers_results[value]
        for classifier_id in classifiers_results:
            result = classifiers_results[classifier_id]
            data.append([value, classifier_id,
                        float_to_string(result["accuracy"]),
                        float_to_string(result["precision"]),
                        float_to_string(result["recall"]),
                        float_to_string(result["f1_score"])])

    elements = []

    cm = 2.54
    doc = SimpleDocTemplate(filename, rightMargin=0, leftMargin=0, topMargin=0.3 * cm, bottomMargin=0)

    table = Table(data)
    table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.25, colors.black), ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black)]))

    # alternative color on the lines
    '''
    data_len = len(data)
    for each in range(1, data_len):
        if each % 2 == 0:
            bg_color = colors.whitesmoke
        else:
            bg_color = colors.lightgrey

        table.setStyle(TableStyle([('BACKGROUND', (0, each), (-1, each), bg_color)]))
    '''

    # Loop through list of lists creating styles for cells with negative value.
    table_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'RIGHT')])
    for row, values, in enumerate(data):
        if row == 0:
            continue
        for column, value in enumerate(values):
            if column < 2:
                continue
            if float(value) < .54:
                table_style.add('BACKGROUND', (column, row), (column, row), colors.red)
    table.setStyle(table_style)

    elements.append(table)
    doc.build(elements)


def make_report_for_value(current_value, library_models, test_vs_pred):
    cm = 2.54
    doc = SimpleDocTemplate("{}.pdf".format(current_value.replace('.','_')), rightMargin=0, leftMargin=0, topMargin=0.3 * cm, bottomMargin=0)

    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph("Alfred : {}".format(current_value), styles['Heading1']))

    from tiase.ml import analysis
    analysis.export_roc_curves(test_vs_pred, "tmp.png", current_value)
    I = Image("tmp.png")
    I.drawHeight = I.drawHeight/3
    I.drawWidth = I.drawWidth/3
    story.append(I)

    for classifier_id in library_models:
        story.append(Paragraph(classifier_id, styles["Normal"]))
        model = library_models[classifier_id]

        model_analysis=model.get_analysis()
        #analysis.export_history("tmp1.png", model_analysis["history"])
        analysis.export_roc_curve(model_analysis["y_test"], model_analysis["y_test_prob"], classifier_id+"tmp2.png")
        analysis.export_confusion_matrix(model_analysis["confusion_matrix"], classifier_id+"tmp3.png")

        I2 = Image(classifier_id+"tmp2.png")
        I2.drawHeight = I2.drawHeight/4
        I2.drawWidth = I2.drawWidth/4

        I3 = Image(classifier_id+"tmp3.png")
        I3.drawHeight = I3.drawHeight/4
        I3.drawWidth = I3.drawWidth/4

        data = [[I2, I3]]
        table = Table(data)
        story.append(table)

    doc.build(story)
