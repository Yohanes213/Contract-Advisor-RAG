import PyPDF2

def extract_text_from_pdf(pdf_path):
    text=""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text+= page.extract_text()

        
    return text


if __name__ == "__main__":
    result = extract_text_from_pdf('scripts/Robinson Advisory.docx.pdf')
    print(len(result))