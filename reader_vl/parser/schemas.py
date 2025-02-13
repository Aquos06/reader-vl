from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import List

from src.advanced_parser.structure.core import StructureBase

class Page(BaseModel):
    page: int
    structure: List[StructureBase]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
class Document(BaseModel):
    filename: str
    filepath: Path
    page: List[Page]
    