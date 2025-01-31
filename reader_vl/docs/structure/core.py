import cv2
import pytesseract
from typing import Optional
import numpy as np
from abc import ABC, abstractmethod

from reader_vl.docs.structure.schemas import ContentType
from reader_vl.docs.structure.registry import register_class, log_info
from reader_vl.llm.client import llmBase

class StructureBase(ABC):
    def __init__(self, coordinate, image: np.ndarray, llm: Optional[llmBase] = None):
        self.coordinate = coordinate
        self.image = image
        self.content = self.get_content(image=image)
        self.llm = llm
        self.secondary_content = self.get_secondary_content(image=image)
        self.metadata = self.get_metadata(image=image)
        
    @log_info
    def get_metadata(image: np.ndarray) -> str:
        return

    @property
    @abstractmethod
    def label()->ContentType:
        return ContentType.NONE
    
    @log_info
    def get_content(self, image: np.ndarray) -> str:
        return pytesseract.image_to_string(image,config="--psm 6")
    
    @log_info
    def get_secondary_content(self, image: np.ndarray) -> str:
        return 
    
    @property
    def labeled_image(self, image: np.ndarray) -> np.ndarray:
        x1,y1,x2,y2 = self.coordinate
        image = cv2.rectangle(image, (x1,y1),(x2,y2), (0,0,125), 1)
        return image
    
@register_class(4)
class Image(StructureBase):
    
    @property
    @abstractmethod
    def label()->ContentType:
        return ContentType.IMAGE
    
    @log_info
    def get_content(self, image: np.ndarray) -> str:
        prompt = ""
        return self.llm.completion(prompt=prompt,image=image)
        
@register_class(2)
class Section(StructureBase):
     @property
     @abstractmethod
     def label()->ContentType:
         return ContentType.SECTION
        
@register_class(7)       
class Table(StructureBase):
    
    @property
    @staticmethod
    def label()->ContentType:
        return ContentType.TABLE
    
    @log_info
    def get_content(self, image: np.ndarray) -> str:
        prompt = "Reformat the table to a markdown type table. Return me only the markdown type table. Do not add or change any data"
        return self.llm.completion(prompt=prompt,image=image)
        
@register_class(0)
class Header(StructureBase):
    def __init__(self, coordinate,image: np.ndarray):
        super().__init__(coordinate,image)
        self.page = self.get_page(image)
        
    @property
    @staticmethod
    def label()->ContentType:
        return ContentType.HEADER
    
    @log_info    
    def get_content(self, image: np.ndarray) -> str:
        prompt = "Return me the text. Do not add or change any data. Return only the text in the image"
        return self.llm.completion(prompt=prompt, image=image)
    
    @log_info    
    def get_page(self, image: np.ndarray) -> str:
        prompt = "If there is page number. Retun the page number, if there is not return None"
        return self.llm.completion(prompt=prompt, image=image)
    
    @log_info    
    def get_metadata(self, image: np.ndarray) -> str:
        prompt = "Return me a metadata according to this image. For example: {'page':50,'title':'FortiOs'}. Do not add or change any text"
        return self.llm.completion(prompt=prompt, image=image)
        
@register_class(1)
class Title(StructureBase):
    
    @property
    @abstractmethod
    def label()->ContentType:
        return ContentType.TITLE
    
    @log_info  
    def get_content(self, image: np.ndarray) -> str:
        prompt = "Return me the text. Do not add or change any data. Return only the text in the image"
        return self.llm.completion(prompt=prompt, image=image)
    
    @property
    def title(self):
        return self.content
    
@register_class(5)
class Footer(StructureBase):
    def __init__(self, coordinate,image: np.ndarray):
        super().__init__(coordinate,image)
        self.page = self.get_page(image)
   
    @property
    @abstractmethod
    def label()->ContentType:
        return ContentType.FOOTER
    
    @log_info
    def get_content(self, image: np.ndarray) -> str:
        prompt = "Return me the text. Do not add or change any data. Return only the text in the image"
        return self.llm.completion(prompt=prompt, image=image)
    
    @log_info
    def get_page(self, image: np.ndarray) -> str:
        prompt = "If there is page number. Retun the page number, if there is not return None"
        return self.llm.completion(prompt=prompt, image=image)
    
    @log_info
    def get_metadata(self, image: np.ndarray) -> str:
        prompt = "Return me a metadata according to this image. For example: {'page':50,'title':'FortiOs'}. Do not add or change any text"
        return self.llm.completion(prompt=prompt, image=image)
    
@register_class(6)
class Chart(StructureBase):
    
    @property
    @abstractmethod
    def label()->ContentType:
        return ContentType.CHART
    
    @log_info
    def get_content(self, image: np.ndarray) -> str:
        prompt = "Describe the chart in this image. Describe the trend, metadata, x-axis, and y-axis. Give me in a form of paragraph. Do not add or change any information, only use the image as your source of information"
        return self.llm.completion(prompt=prompt, image=image)

@register_class(8)
class Reference(StructureBase):
    @property
    @abstractmethod
    def label()->ContentType:
        return ContentType.REFERENCE

@register_class(9)
class FigureCaption(StructureBase):
    @property
    @abstractmethod
    def label()->ContentType:
        return ContentType.FIGURECAPTION
    
@register_class(10)
class TableCaption(StructureBase):
    @property
    @abstractmethod
    def label()->ContentType:
        return ContentType.TABLECAPTION

@register_class(11)
class Equation(StructureBase):
    @property
    @abstractmethod
    def label()->ContentType:
        return ContentType.EQUATION

    @log_info
    def get_content(self, image: np.ndarray) -> str:
        prompt = "Return me the equation given in the image. Explain the given equation in a paragraph. Do not change or add any text in when writting the equation. The return format will be: \n Equation: '....' \n Description: '..."
        return self.llm.completion(prompt=prompt, image=image)

@register_class(3)
class List(StructureBase):
    @property
    @abstractmethod
    def label()->ContentType:
        return ContentType.LIST