from fpdf import FPDF

def generate_pdf(text):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0,10,text)

    pdf.output("report.pdf")

    return "report.pdf"