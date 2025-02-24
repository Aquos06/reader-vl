from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from pdf2image import convert_from_bytes


def open_file(file_path: Optional[Union[Path, str]]) -> bytes:
    """
    Opens a file and returns its content as bytes.

    Args:
      file_path: The path to the file. Can be a pathlib.Path object or a string.

    Returns:
      The content of the file as bytes.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    with file_path.open("rb") as file:
        return file.read()


def pdf2image(pdf_bytes: bytes) -> List[np.ndarray]:
    """
    Converts a PDF file (given as bytes) to a list of images.

    Args:
        pdf_bytes: The PDF file content as bytes.

    Returns:
        A list of NumPy arrays, where each array represents an image.
    """
    images = convert_from_bytes(pdf_bytes)
    return [np.array(image) for image in images]
