import pdfplumber

class CorruptedPDFError(Exception):
    """Raised when the PDF cannot be opened or parsed."""
    pass

class EmptyPDFError(Exception):
    """Raised when the PDF has no pages or contains no extractable text."""
    pass

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from a PDF file-like object using pdfplumber.
    
    Args:
        pdf_file: File-like object or file path.
        
    Returns:
        str: Extracted clean text from the PDF.
        
    Raises:
        EmptyPDFError: If PDF is empty or has no extractable text.
        CorruptedPDFError: If there was a parsing issue.
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            if not pdf.pages:
                raise EmptyPDFError("The uploaded PDF report contains no pages.")
            
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            
            full_text = "\n".join(pages_text).strip()
            if not full_text:
                raise EmptyPDFError(
                    "The PDF appears to have no extractable text. "
                    "Make sure it is not a scanned image PDF."
                )
            return full_text
    except Exception as e:
        if isinstance(e, EmptyPDFError):
            raise
        raise CorruptedPDFError(f"Could not read PDF file: {e}")
