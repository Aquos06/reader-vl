import os
from pathlib import Path
from typing import Optional, Union

from docx2pdf import convert

from reader_vl.docs.schemas import Document
from reader_vl.docs.structure.schemas import ContentType
from reader_vl.docs.utils import pdf2image
from reader_vl.document_reader.base import ReaderBase
from reader_vl.llm.client import llmBase


class DocxReader(ReaderBase):
    def __init__(
        self,
        llm: llmBase,
        file_path: Optional[Union[Path, str]] = None,
        file_bytes: Optional[bytes] = None,
        metadata: Optional[dict] = None,
        yolo_paremeters: Optional[dict] = {},
        verbose: Optional[bool] = True,
        auto_parse: Optional[bool] = True,
        structure_custom_prompt: Optional[dict[ContentType, str]] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            file_path=file_path,
            file_bytes=file_bytes,
            metadata=metadata,
            yolo_parameters=yolo_paremeters,
            verbose=verbose,
            auto_parse=auto_parse,
            structure_custom_prompt=structure_custom_prompt,
        )

        if self.file_path and self.file_path.suffix != ".docx":
            raise ValueError("DocxReader only support Docx files")

        self.pdf_bytes = self._convert_docx_bytes_to_pdf(self.file_bytes)

    def _convert_docx_bytes_to_pdf(self, docx_bytes, file_path=None):
        temp_pdf_path = "output.pdf"
        if not file_path:
            with open("temp.docx", "wb") as temp_file:
                temp_file.write(docx_bytes)

            file_path = "temp.docx"

        convert(file_path, temp_pdf_path)

        with open(temp_pdf_path, "rb") as file:
            pdf_bytes = file.read()

        os.remove(temp_pdf_path)
        return pdf_bytes

    def parse(self) -> Document:
        self.images = pdf2image(self.pdf_bytes)
        return self._parse(images=self.images)

    async def aparse(self) -> Document:
        self.images = pdf2image(self.pdf_bytes)
        return await self._aparse(self.images)
