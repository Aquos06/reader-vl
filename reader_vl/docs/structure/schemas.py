from enum import Enum


class ContentType(Enum):
    IMAGE = "Image"
    SECTION = "Section"
    TABLE = "Table"
    HEADER = "Header"
    TITLE = "Title"
    FOOTER = "Footer"
    CHART = "Chart"
    REFERENCE = "Reference"
    FIGURECAPTION = "FigureCaption"
    TABLECAPTION = "TableCaption"
    EQUATION = "Equation"
    LIST = "List"
    NONE = None
