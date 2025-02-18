import cv2
import base64
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Union, Optional, List

from reader_vl.docs.yolo import YOLO
from reader_vl.llm.client import llmBase
from reader_vl.docs.utils import open_file, pdf2image
from reader_vl.docs.structure.core import StructureBase
from reader_vl.docs.structure.schemas import ContentType
from reader_vl.docs.schemas import Document, Page, Component
from reader_vl.docs.structure.registry import CLASS_REGISTRY
from reader_vl.models.utils import get_models_path

logging.basicConfig(level=logging.INFO)

class DocReader:
    def __init__(
            self,
            llm: llmBase,
            file_path: Optional[Union[Path, str]] = None,
            file_bytes: Optional[bytes] = None,
            metadata: Optional[dict] = None,
            yolo_parameters: Optional[dict] = {},
            verbose: Optional[bool] = True,
            failed_image_path: Optional[Union[str, Path]] = Path("./failed.jpg"),
            auto_parse: Optional[bool] = True,
        )->None:
        self.check_arguments(file_path, file_bytes)

        WEIGHT_PATHS = get_models_path()
        
        if file_path:
            self.metadata = {
                "file_path": file_path,
                "file_name": file_path.name,
                **(metadata or  {})
            }
            self.file_bytes = open_file(file_path)
        
        self.llm = llm
        self.verbose = verbose
        self.yolo = YOLO(weight_path=WEIGHT_PATHS, **yolo_parameters)
        self.failed_image_path = failed_image_path
        self.file_name = file_path.name if file_path else None
        self.file_path = file_path
        self.file_bytes = file_bytes
        
        self.parsed_document = None
        if auto_parse:
            self.parsed_document:Document = self.parse()
        
        
    def check_arguments(file_path: Optional[Union[Path, str]], file_bytes: bytes) -> None:
        if file_bytes and file_path:
            raise ValueError("file_path and file_bytes cannot be input at the same time") 
        if not file_bytes and not file_path:
            raise ValueError("file_path and file_bytes need to be given during the initialization")
        if file_path and file_path.suffix != ".pdf":
                raise ValueError("Only PDF files are currently supported") 
        
    def _parse(self, images:List[np.ndarray])->Document:
        components: List[Page] = []
        
        if self.verbose:
            image_iterator = tqdm(enumerate(images), desc="Processing Image")
        else:
            image_iterator = enumerate(images)
            
        for index, image in image_iterator:
            results = self.yolo(image)
            child_components:List[Component] = []
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
                    
                        component:StructureBase = CLASS_REGISTRY[box_class](coordinate,self.llm)
                        child_components.append(
                            Component(
                                content=component.content,
                                coordinate=component.coordinate,
                                secondary_content=component.secondary_content,
                                metadata=component.metadata,
                                component_type=component.label
                            )
                        )
                        
                    except Exception as e:
                        logging.error(f'error: {e}, class: {box_class} \n coordinate: ({x1},{y1},{x2},{y2})', exc_info=True)
                        cv2.imwrite(self.failed_image_path, cut_image)
                        logging.info(f'save image to {self.failed_image_path}')
                        continue
                    
            components.append(Page(
                page=index,
                component=child_components,
                metadata={} #TODO add metadata
            ))
            
        return Document(
            filename=self.file_name,
            filepath=self.file_path,
            page=components,
            metadata={}
        )
    
    def parse(self)->Document:
        images = pdf2image(self.file_bytes)
        return self._parse(images=images)
    
    async def aparse(self)->Document:
        images = pdf2image(self.file_bytes)
        return self._parse(images=images)
    
    def export_to_json(self) -> List[dict]:
        if self.parsed_document == None:
            self.parsed_document = self.parse()

        json_list: List[dict] = []
        for page in self.parsed_document.page:
            for component in page.component:
                json_list.append(component.model_dump_json())

        return json_list
    
    def export_to_markdown(self, ignore_footer: Optional[bool] = True) -> str:
        if self.parsed_document == None:
            self.parsed_document = self.parse()

        markdown_output = "" 
        for page in self.parsed_document.page:
            for component in page.component:
                if component.component_type == ContentType.HEADER:
                    markdown_output += f"### {component.content}\n\n"
                elif component.component_type == ContentType.SECTION:
                    markdown_output += f"{component.content} \n"
                elif component.component_type == ContentType.TABLE:
                    markdown_output += self._format_table(component.content)
                elif component.component_type == ContentType.LIST:
                    markdown_output += self._format_list(component.content)
                elif component.component_type == ContentType.IMAGE:
                    markdown_output += self._format_image(component.content)
                elif component.component_type == ContentType.EQUATION:
                    markdown_output += f"$${component.content}$$ \n\n"
                elif component.component_type == ContentType.FIGURECAPTION:
                    markdown_output += f"{component.component_type} \n"
                elif component.component_type == ContentType.TITLE:
                    markdown_output += f"# {component.content}\n"
                elif component.component_type == ContentType.TABLECAPTION:
                    markdown_output += f"{component.content}\n"
                elif component.component_type == ContentType.REFERENCE:
                    markdown_output += f"{component.content}\n"
                elif component.component_type == ContentType.CHART:
                    markdown_output += self._format_chart(component.content)
                elif component.component_type == ContentType.FOOTER and not ignore_footer:
                    markdown_output += f"{component.content}\n"
                    
        return

    def _format_table(self, table_data: str) -> str:
        if not isinstance(table_data, list) or not all(isinstance(row, list) for row in table_data):
            return "Error: Invalid table format" # Handle cases where table format is wrong
        markdown_table = ""
        for i, row in enumerate(table_data):
            markdown_table += "| " + " | ".join(map(str, row)) + " |\n"
            if i == 0:  # Add header separator
                markdown_table += "| " + " | ".join(["---"] * len(row)) + " |\n"
        return markdown_table
    
    def _format_list(self, list_data: str) -> str:
        if not isinstance(list_data, list):
            return "Error: Invalid list format"
        markdown_list = ""
        for item in list_data:
            markdown_list += f"- {item}\n"
        return markdown_list
    
    def _format_image(self, image_data) -> str:
        try:
            encoded_string = base64.b64encode(image_data).decode("utf-8")
            return encoded_string # Assumes PNG change if needed
        except Exception as e:
            logging.error(f"Error embedding image: {e}")
            return "Error displaying image.\n\n"
 
    def _format_chart(self, chart_data) -> str:
        try:
            encoded_string = base64.b64encode(chart_data).decode("utf-8")
            return encoded_string # Assumes PNG change if needed
        except Exception as e:
            logging.error(f"Error embedding image: {e}")
            return "Error displaying image.\n\n"