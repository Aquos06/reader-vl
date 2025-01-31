from abc import ABC, abstractmethod
from typing import List

from llm.schemas import CompletionResponse, ChatMessage, ChatCompletionResponse

class llmBase(ABC):
    def __init__(self, url: str) -> None:
        self.url = url

    @abstractmethod
    def completion(self, prompt: str, *args, **kwargs) -> CompletionResponse:
        pass
    
    @abstractmethod
    def acompletion(self, prompt: str, *args, **kwargs) -> CompletionResponse:
        pass
    
    @abstractmethod
    def chat(self, message: List[ChatMessage], *args, **kwargs) -> ChatCompletionResponse:
        pass
    
    @abstractmethod
    def achat(self, message: List[ChatMessage], *args, **kwargs) -> ChatCompletionResponse:
        pass