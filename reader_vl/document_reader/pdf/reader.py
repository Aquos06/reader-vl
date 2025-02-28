from pathlib import Path
from typing import Optional, Union

from reader_vl.docs.structure.schemas import ContentType
from reader_vl.document_reader.base import ReaderBase
from reader_vl.llm.client import llmBase


class PDFReader(ReaderBase):
    def __init__(
        self,
        llm: llmBase,
        file_path: Optional[Union[Path, str]] = None,
        file_bytes: Optional[bytes] = None,
        metadata: Optional[dict] = None,
        yolo_parameters: Optional[dict] = {},
        verbose: Optional[bool] = True,
        auto_parse: Optional[bool] = True,
        structure_custom_prompt: Optional[dict[ContentType, str]] = None,
    ):
        super().__init__(
            llm=llm,
            file_path=file_path,
            file_bytes=file_bytes,
            metadata=metadata,
            yolo_parameters=yolo_parameters,
            verbose=verbose,
            auto_parse=auto_parse,
            structure_custom_prompt=structure_custom_prompt,
        )
