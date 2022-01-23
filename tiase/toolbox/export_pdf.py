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

def format_data_for_table(values_classifiers_results):
    data = []
    data.append(["classifier id", "Accuracy", "Precision", "Recall", "F1 score"])
    for classifier_id in values_classifiers_results:
        data.append([classifier_id,
                    float_to_string(values_classifiers_results[classifier_id]["accuracy"]),
                    float_to_string(values_classifiers_results[classifier_id]["precision"]),
                    float_to_string(values_classifiers_results[classifier_id]["recall"]),
                    float_to_string(values_classifiers_results[classifier_id]["f1_score"])])

    return data

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

    # details for each value
    for value in report["values_classifiers_results"]:
        elements.append(PageBreak()) # break
        elements.append(Paragraph("{}".format(value), styles['Heading1']))

        classifiers_results = report["values_classifiers_results"][value]
        roc_curves_filename = classifiers_results.get("roc_curves_filename", "")
        if roc_curves_filename:
            elements.append(get_image(roc_curves_filename))

        data = format_data_for_table(report["values_classifiers_results"][value]["classifiers"])

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

        # Loop through data creating styles for cells
        table_style = TableStyle([('ALIGN', (0, 0), (-1, -1), 'RIGHT')])
        for row, current_values, in enumerate(data):
            if row == 0:
                continue
            for column, current_value in enumerate(current_values):
                if column < 1:
                    continue
                if float(current_value) < .55:
                    table_style.add('BACKGROUND', (column, row), (column, row), colors.red)
                if float(current_value) >= .6:
                    table_style.add('BACKGROUND', (column, row), (column, row), colors.green)
        table.setStyle(table_style)

        elements.append(table)

        for key in report["values_classifiers_results"][value]["classifiers"]:
            elements.append(Paragraph("{}".format(key), styles['Heading2']))
            classifier_model_analysis = report["values_classifiers_results"][value]["classifiers"][key]
            I2 = get_image(classifier_model_analysis["roc_curve_filename"])
            I3 = get_image(classifier_model_analysis["confusion_matrix_fiename"])

            data = [[I2, I3]]
            table = Table(data)
            elements.append(table)

    doc.build(elements)
