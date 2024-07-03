import PyPDF2
from pypdf import PdfReader

# def extract_text_from_pdf(pdf_path):
#     text=""
#     with open(pdf_path, "rb") as file:
#         reader = PyPDF2.PdfReader(file)

#         num_pages = len(reader.pages)
#         for page_num in range(num_pages):
#             page = reader.pages[page_num]
#             text+= page.extract_text()

        
#     return text

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    pdf_texts = [text for text in pdf_texts if text]

    return pdf_texts

        
    #return text


if __name__ == "__main__":
    result = extract_text_from_pdf('scripts/Robinson Advisory.docx.pdf')
    print(result[0])