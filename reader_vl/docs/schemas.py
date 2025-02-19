import numpy as np
from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import List, Dict, Optional

from reader_vl.docs.structure.core import StructureBase
from reader_vl.docs.structure.schemas import ContentType

class Component(BaseModel):
    content: str
    coordinate: Dict[str, float]
    secondary_content: Optional[str] = None
    metadata: Optional[Dict] = {}
    component_type: ContentType
    image: Optional[np.ndarray] = None

class Page(BaseModel):
    page: int
    component: List[Component]
    metadata: Optional[Dict] = {}
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
class Document(BaseModel):
    filename: str
    filepath: Path
    page: List[Page]
    metadata: Optional[Dict] = []