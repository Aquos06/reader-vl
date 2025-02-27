import logging
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
import pytesseract

from reader_vl.docs.structure.prompt import (
    CHART_PROMPT,
    EQUATION_PROMPT,
    FOOTER_PROMPT,
    HEADER_PROMPT,
    IMAGE_PROMPT,
    TABLE_PROMPT,
    TITLE_PROMPT,
)
from reader_vl.docs.structure.registry import log_info, register_class
from reader_vl.docs.structure.schemas import ContentType
from reader_vl.llm.client import llmBase
from reader_vl.llm.schemas import ChatCompletionResponse

logging.basicConfig(level=logging.INFO)


class StructureBase(ABC):
    def __init__(
        self,
        coordinate,
        image: np.ndarray,
        llm: Optional[llmBase] = None,
        prompt: Optional[str] = None,
        content: Optional[str] = None,
    ):
        self.coordinate = coordinate
        self.image = image

        self.llm = llm
        self.secondary_content = self.get_secondary_content(image=image)
        self.metadata = self.get_metadata(image=image)
        self.prompt = prompt
        self.content = content if content else self.get_content(image=image)

    @classmethod
    async def create(
        cls, coordinate, image: np.ndarray, llm=None, prompt=None, is_async=False
    ):
        """Asynchronous factory method"""
        content = await cls.get_content(image) if is_async else cls.get_content(image)
        return cls(coordinate, image, llm, prompt, is_async, content=content)

    @log_info
    def get_metadata(self, image: np.ndarray) -> str:
        return

    @property
    @abstractmethod
    def label(self) -> ContentType:
        return ContentType.NONE

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        return pytesseract.image_to_string(image, config="--psm 6")

    @log_info
    async def aget_content(self, image: np.ndarray) -> str:
        return self.get_content(image=image)

    @log_info
    def get_secondary_content(self, image: np.ndarray) -> str:
        return

    @property
    def labeled_image(self, image: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.coordinate
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 125), 1)
        return image

    def _get_content_from_llm(self, response: ChatCompletionResponse) -> str:
        return response.choices[0].message.content


@register_class(4)
class Image(StructureBase):
    def __init__(
        self,
        coordinate,
        image: np.ndarray,
        llm: Optional[llmBase] = None,
        prompt: Optional[str] = IMAGE_PROMPT,
    ):
        super().__init__(coordinate=coordinate, image=image, llm=llm, prompt=prompt)

    @property
    def label(self) -> ContentType:
        return ContentType.IMAGE

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        response = self.llm.chat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    async def aget_content(self, image: np.ndarray) -> str:
        response = await self.llm.achat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)


@register_class(2)
class Section(StructureBase):
    @property
    def label(self) -> ContentType:
        return ContentType.SECTION


@register_class(7)
class Table(StructureBase):
    def __init__(
        self,
        coordinate,
        image: np.ndarray,
        llm: Optional[llmBase] = None,
        prompt: Optional[str] = TABLE_PROMPT,
    ):
        super().__init__(coordinate=coordinate, image=image, llm=llm, prompt=prompt)

    @property
    def label(self) -> ContentType:
        return ContentType.TABLE

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        response = self.llm.chat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    async def aget_content(self, image: np.ndarray) -> str:
        response = await self.llm.achat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)


@register_class(0)
class Header(StructureBase):
    def __init__(
        self,
        coordinate,
        image: np.ndarray,
        llm: Optional[llmBase] = None,
        prompt: Optional[str] = HEADER_PROMPT,
    ):
        super().__init__(coordinate=coordinate, image=image, llm=llm, prompt=prompt)
        self.page = self.get_page(image)

    @property
    def label(self) -> ContentType:
        return ContentType.HEADER

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        response = self.llm.chat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    async def aget_content(self, image: np.ndarray) -> str:
        response = await self.llm.achat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)


@register_class(1)
class Title(StructureBase):
    def __init__(
        self,
        coordinate,
        image: np.ndarray,
        llm: Optional[llmBase] = None,
        prompt: Optional[str] = TITLE_PROMPT,
    ):
        super().__init__(coordinate=coordinate, image=image, llm=llm, prompt=prompt)

    @property
    def label(self) -> ContentType:
        return ContentType.TITLE

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        response = self.llm.chat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    async def aget_content(self, image: np.ndarray) -> str:
        response = await self.llm.achat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @property
    def title(self):
        return self.content


@register_class(5)
class Footer(StructureBase):
    def __init__(
        self,
        coordinate,
        image: np.ndarray,
        llm: Optional[llmBase] = None,
        prompt: Optional[str] = FOOTER_PROMPT,
    ):
        super().__init__(coordinate=coordinate, image=image, llm=llm, prompt=prompt)
        self.page = self.get_page(image)

    @property
    def label(self) -> ContentType:
        return ContentType.FOOTER

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        response = self.llm.chat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    async def aget_content(self, image: np.ndarray) -> str:
        response = await self.llm.achat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)


@register_class(6)
class Chart(StructureBase):
    def __init__(
        self,
        coordinate,
        image: np.ndarray,
        llm: Optional[llmBase] = None,
        prompt: Optional[str] = CHART_PROMPT,
    ):
        super().__init__(coordinate=coordinate, image=image, llm=llm, prompt=prompt)

    @property
    def label(self) -> ContentType:
        return ContentType.CHART

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        response = self.llm.chat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    async def aget_content(self, image: np.ndarray) -> str:
        response = await self.llm.achat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)


@register_class(8)
class Reference(StructureBase):
    @property
    def label(self) -> ContentType:
        return ContentType.REFERENCE


@register_class(9)
class FigureCaption(StructureBase):
    @property
    def label(self) -> ContentType:
        return ContentType.FIGURECAPTION


@register_class(10)
class TableCaption(StructureBase):
    @property
    def label(self) -> ContentType:
        return ContentType.TABLECAPTION


@register_class(11)
class Equation(StructureBase):
    def __init__(
        self,
        coordinate,
        image: np.ndarray,
        llm: Optional[llmBase] = None,
        prompt: Optional[str] = EQUATION_PROMPT,
    ):
        super().__init__(coordinate=coordinate, image=image, llm=llm, prompt=prompt)

    @property
    def label(self) -> ContentType:
        return ContentType.EQUATION

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        response = self.llm.chat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    async def aget_content(self, image: np.ndarray) -> str:
        response = await self.llm.achat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)


@register_class(3)
class List(StructureBase):
    @property
    def label(self) -> ContentType:
        return ContentType.LIST
