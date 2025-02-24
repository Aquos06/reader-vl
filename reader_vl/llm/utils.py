import base64

import cv2


def encode_image(image) -> str:
    """
    Encodes an image into a base64 string.

    Args:
        image: The image to encode (e.g., a NumPy array).

    Returns:
        A base64-encoded string representing the image.
    """
    _, buffer = cv2.imencode(".png", image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return image_base64
