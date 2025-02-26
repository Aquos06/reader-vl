import logging
from functools import wraps
from typing import Any, Dict

from reader_vl.docs.structure.schemas import ContentType

CLASS_REGISTRY: Dict[ContentType, Any] = {}
logging.basicConfig(level=logging.INFO)


def register_class(key):
    def decorator(cls):
        CLASS_REGISTRY[key] = cls
        return cls

    return decorator


def log_info(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.__class__.__name__
        result = func(self, *args, **kwargs)
        return result

    return wrapper
