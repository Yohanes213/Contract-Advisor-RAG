from pypdf import PdfReader
from scripts.logger import logger


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file located at `pdf_path`.

    Args:
    - pdf_path (str): Path to the PDF file.

    Returns:
    - list: List of text extracted from the PDF pages.
           Returns an empty list if there is any error during extraction.
    """
    try:
        reader = PdfReader(pdf_path)
        pdf_texts = [p.extract_text() for p in reader.pages if p.extract_text()]
        logger.info(f"Extracting text from PDF '{pdf_path}' successfully completed")
        return pdf_texts
    except Exception as e:
        logger.error(f"Error extracting text from PDF '{pdf_path}': {str(e)}")
        return []


if __name__ == "__main__":
    result = extract_text_from_pdf("data/Robinson Advisory.docx.pdf")
    print(result)
