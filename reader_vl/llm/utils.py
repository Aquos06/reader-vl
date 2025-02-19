import base64

import cv2


def encode_image(image) -> str:
    _, buffer = cv2.imencode(".png", image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return image_base64
