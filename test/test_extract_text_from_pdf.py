import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.document_extraction.document_extractor import extract_text_from_pdf


def test_extract_text_from_valid_pdf():
    """Tests extraction from a valid PDF file."""
    pdf_path = "test/sample.pdf" 

    extracted_text = extract_text_from_pdf(pdf_path)

    # Assertions
    assert isinstance(extracted_text, list)
    assert len(extracted_text) > 0 

