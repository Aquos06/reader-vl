import cv2
import logging
import numpy as np

from tqdm import tqdm
from pdf2image import convert_from_bytes
from typing import List
from pathlib import Path


from src.advanced_parser.const import WEIGTH_PATH
from src.advanced_parser.yolo import YOLO
from src.advanced_parser.structure.registry import CLASS_REGISTRY
from src.advanced_parser.parser.schemas import Page, Document

logging.basicConfig(level=logging.INFO)


class PDFParser:
    def __init__(self, pdf_bytes=None, filename: str="", file_path: Path="") -> None:
        self.pdf_bytes = pdf_bytes
        self.file_name = filename
        self.file_path = file_path
        self.pdf_images = self.convert_pdf_to_images(pdf_bytes=self.pdf_bytes)
        self.model = YOLO(WEIGTH_PATH)
    
    def __call__(self) -> Document|None:
        components: List[Page] = []
        for index, image in tqdm(enumerate(self.pdf_images),desc="Processing image"):
            results = self.model(image)
            logging.info(image.shape)
            child_components = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        x1,y1,x2,y2 = box.xyxy[0]
                        coordinate = (x1,y1,x2,y2)
                        box_class = int(box.cls[0])
                        if abs(y2-y1) < 10:
                            continue
                        cut_image = image[int(y1):int(y2),int(x1):int(x2)]
                        
                        child_components.append(CLASS_REGISTRY[box_class](coordinate,cut_image))
                    except Exception as e:
                        logging.error(f'error: {e}, class: {box_class}, \n coordinate: ({x1},{y1},{x2},{y2})', exc_info=True)
                        cv2.imwrite('/home/hansen/workspace/parser/error.jpg',cut_image)
                        logging.info(f'Saved images')
                        return
                   
            components.append(Page(
                page=index,
                structure=child_components
            ))
            
        return Document(
            filename=self.file_name,
            filepath=self.file_path,
            page=components
        )
    
    def convert_pdf_to_images(self, pdf_bytes) -> List[np.ndarray]:
        images = convert_from_bytes(pdf_bytes)
        return [np.array(image) for image in images]
    
    def parse_from_file_path(self):
        with open(self.file_name, 'rb') as file:
            self.pdf_bytes = file.read()
            self.__call__()
    