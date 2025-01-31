import requests
import numpy as np

from src.advanced_parser.const import VLLM_URL
from src.advanced_parser.llm.utils import encode_image

def call_vllm(prompt: str, image: np.ndarray ) -> str:
    encoded_image = encode_image(image=image)
    data = {
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }],
        "model": "/mnt/model",
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 50,
        "stop":["<|eot_id|>"]
    }
    
    response = requests.post(url=VLLM_URL,json=data)
    response.raise_for_status()
    response = response.json()
    
    return response['choices'][0]['message']['content']