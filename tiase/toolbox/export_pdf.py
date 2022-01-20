from reportlab.lib import colors,utils
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle

cm = 2.54

def float_to_string(value, n_decimals=4):
    return '{:.{n_decimals}f}'.format(value, n_decimals=n_decimals)

def get_image(path, width=70*cm):
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))

def make_report(report, filename):
    print("\U0001F4D6 [EXPORT PDF] {}".format(filename))
    doc = SimpleDocTemplate(filename, rightMargin=0, leftMargin=0, topMargin=0.3 * cm, bottomMargin=0)
    styles = getSampleStyleSheet()
    
    elements = []

    # header
    elements.append(Paragraph("Alfred report", styles['Heading1']))

    # xmlfile
    with open(report["xmlfile"], "r") as file:
        xmlcontent = file.read()
        xmlcontent = xmlcontent.replace('<','&lt;')
        xmlcontent = xmlcontent.replace('>','&gt;')
        xmlcontent = xmlcontent.replace(' ','&nbsp;')
        xmlcontent = xmlcontent.replace('\n','<br/>')

        elements.append(Paragraph("Input :", styles['Heading2']))
        elements.append(Paragraph(xmlcontent, ParagraphStyle('xml', fontName="Courier", fontSize=6)))

    # break
    elements.append(PageBreak())

    # format data to be treated by the reportlab Table
    data = []
    data.append(["value", "classifier id", "Accuracy", "Precision", "Recall", "F1 score"])
    values_classifiers_results = report["values_classifiers_results"]
    for value in values_classifiers_results:
        classifiers_results = values_classifiers_results[value]["classifiers"]
        for classifier_id in classifiers_results:
            result = classifiers_results[classifier_id]
            data.append([value, classifier_id,
                        float_to_string(result["accuracy"]),
                        float_to_string(result["precision"]),
                        float_to_string(result["recall"]),
                        float_to_string(result["f1_score"])])


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
            if float(value) < .55:
                table_style.add('BACKGROUND', (column, row), (column, row), colors.red)
            if float(value) >= .6:
                table_style.add('BACKGROUND', (column, row), (column, row), colors.green)
    table.setStyle(table_style)
    elements.append(table)


    # details for each value
    for value in values_classifiers_results:
        classifiers_results = values_classifiers_results[value]
        roc_curves_filename = classifiers_results.get("roc_curves_filename", "")
        if roc_curves_filename:
            elements.append(get_image(roc_curves_filename))


    doc.build(elements)


def make_report_for_value(current_value, library_models, test_vs_pred):
    
    doc = SimpleDocTemplate("{}.pdf".format(current_value.replace('.','_')), rightMargin=0, leftMargin=0, topMargin=0.3 * cm, bottomMargin=0)

    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph("Alfred : {}".format(current_value), styles['Heading1']))

    from tiase.ml import analysis
    analysis.export_roc_curves(test_vs_pred, "tmp.png", current_value)
    story.append(get_image("tmp.png"))

    for classifier_id in library_models:
        story.append(Paragraph(classifier_id, styles["Normal"]))
        model = library_models[classifier_id]

        model_analysis=model.get_analysis()
        #analysis.export_history("tmp1.png", model_analysis["history"])
        analysis.export_roc_curve(model_analysis["y_test"], model_analysis["y_test_prob"], classifier_id+"tmp2.png")
        analysis.export_confusion_matrix(model_analysis["confusion_matrix"], classifier_id+"tmp3.png")

        I2 = get_image(classifier_id+"tmp2.png")
        I3 = get_image(classifier_id+"tmp3.png")

        data = [[I2, I3]]
        table = Table(data)
        story.append(table)

    doc.build(story)
