import cv2
import base64

def encode_image(image) -> str:
    _, buffer = cv2.imencode('.png',image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64