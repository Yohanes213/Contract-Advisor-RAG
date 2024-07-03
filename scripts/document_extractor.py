from pypdf import PdfReader
from logger import logger  # Import logger from logger.py

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        pdf_texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
        logger.info(f"Extracting text from PDF '{pdf_path}' successfully completed")
        return pdf_texts
    except Exception as e:
        logger.error(f"Error extracting text from PDF '{pdf_path}': {str(e)}")
        return []



if __name__ == "__main__":
    result = extract_text_from_pdf('scripts/Robinson Advisory.docx.pdf')
    print(result[0])