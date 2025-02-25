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
    def get_metadata(image: np.ndarray) -> str:
        return

    @property
    @abstractmethod
    def label() -> ContentType:
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
        return response.choices[0].message.content[0].text


@register_class(4)
class Image(StructureBase):
    def __init__(self, coordinate, image: np.ndarray, llm: Optional[llmBase] = None):
        super().__init__(
            coordinate=coordinate, image=image, llm=llm, prompt=IMAGE_PROMPT
        )

    @property
    @abstractmethod
    def label() -> ContentType:
        return ContentType.IMAGE

    @abstractmethod
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

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
    @abstractmethod
    def label() -> ContentType:
        return ContentType.SECTION


@register_class(7)
class Table(StructureBase):
    def __init__(self, coordinate, image: np.ndarray, llm: Optional[llmBase] = None):
        super().__init__(
            coordinate=coordinate, image=image, llm=llm, prompt=TABLE_PROMPT
        )

    @property
    @staticmethod
    def label() -> ContentType:
        return ContentType.TABLE

    @abstractmethod
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

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
    def __init__(self, coordinate, image: np.ndarray, llm: Optional[llmBase] = None):
        super().__init__(
            coordinate=coordinate, image=image, llm=llm, prompt=HEADER_PROMPT
        )
        self.page = self.get_page(image)

    @property
    @staticmethod
    def label() -> ContentType:
        return ContentType.HEADER

    @abstractmethod
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        response = self.llm.chat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    async def aget_content(self, image: np.ndarray) -> str:
        response = await self.llm.achat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    def get_page(self, image: np.ndarray) -> str:
        prompt = "If there is page number. Retun the page number, if there is not return None"
        return self.llm.chat(prompt=prompt, image=image)

    @log_info
    def get_metadata(self, image: np.ndarray) -> str:
        prompt = "Return me a metadata according to this image. For example: {'page':50,'title':'FortiOs'}. Do not add or change any text"
        return self.llm.chat(prompt=prompt, image=image)


@register_class(1)
class Title(StructureBase):
    def __init__(self, coordinate, image: np.ndarray, llm: Optional[llmBase] = None):
        super().__init__(
            coordinate=coordinate, image=image, llm=llm, prompt=TITLE_PROMPT
        )

    @property
    @abstractmethod
    def label() -> ContentType:
        return ContentType.TITLE

    @abstractmethod
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

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
    def __init__(self, coordinate, image: np.ndarray, llm: Optional[llmBase] = None):
        super().__init__(
            coordinate=coordinate, image=image, llm=llm, prompt=FOOTER_PROMPT
        )
        self.page = self.get_page(image)

    @property
    @abstractmethod
    def label() -> ContentType:
        return ContentType.FOOTER

    @abstractmethod
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        response = self.llm.chat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    async def aget_content(self, image: np.ndarray) -> str:
        response = await self.llm.achat(prompt=self.prompt, image=image)
        return self._get_content_from_llm(response=response)

    @log_info
    def get_page(self, image: np.ndarray) -> str:
        prompt = "If there is page number. Retun the page number, if there is not return None"
        return self.llm.chat(prompt=prompt, image=image)

    @log_info
    def get_metadata(self, image: np.ndarray) -> str:
        prompt = "Return me a metadata according to this image. For example: {'page':50,'title':'FortiOs'}. Do not add or change any text"
        return self.llm.chat(prompt=prompt, image=image)


@register_class(6)
class Chart(StructureBase):
    def __init__(self, coordinate, image: np.ndarray, llm: Optional[llmBase] = None):
        super().__init__(
            coordinate=coordinate, image=image, llm=llm, prompt=CHART_PROMPT
        )

    @property
    @abstractmethod
    def label() -> ContentType:
        return ContentType.CHART

    @abstractmethod
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

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
    @abstractmethod
    def label() -> ContentType:
        return ContentType.REFERENCE


@register_class(9)
class FigureCaption(StructureBase):
    @property
    @abstractmethod
    def label() -> ContentType:
        return ContentType.FIGURECAPTION


@register_class(10)
class TableCaption(StructureBase):
    @property
    @abstractmethod
    def label() -> ContentType:
        return ContentType.TABLECAPTION


@register_class(11)
class Equation(StructureBase):
    def __init__(self, coordinate, image: np.ndarray, llm: Optional[llmBase] = None):
        super().__init__(
            coordinate=coordinate, image=image, llm=llm, prompt=EQUATION_PROMPT
        )

    @property
    @abstractmethod
    def label() -> ContentType:
        return ContentType.EQUATION

    @abstractmethod
    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

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
    @abstractmethod
    def label() -> ContentType:
        return ContentType.LIST
