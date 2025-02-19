from pathlib import Path

from docx2pdf import convert
from src.advanced_parser.parser.pdf import PDFParser


class DocsParser:
    def __init__(self, docx_bytes, filename: str, file_path: Path) -> None:
        self.docx_bytes = docx_bytes
        self.filename = filename
        self.file_path = file_path
        self.pdf_bytes = self.convert_docx_bytes_to_pdf(self.docx_bytes)
        self.pdf_parser = PDFParser(
            pdf_bytes=self.pdf_bytes, filename=filename, file_path=file_path
        )

    def convert_docx_bytes_to_pdf(self, docx_bytes):
        with open("temp.docx", "wb") as temp_file:
            temp_file.write(docx_bytes)

        convert("temp.docx", "output.pdf")

        with open("output.pdf", "rb") as file:
            return file.read()

    def Docx2PdfParser(self):
        return self.pdf_parser
