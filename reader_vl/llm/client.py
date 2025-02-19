from abc import ABC, abstractmethod
from typing import Generator, List, Optional

from llm.schemas import ChatCompletionResponse, ChatMessage, CompletionResponse


class llmBase(ABC):
    def __init__(self, url: str, model: str, max_tokens: Optional[int]) -> None:
        self.url = url
        self.model = model
        self.max_tokens = max_tokens

    @abstractmethod
    def completion(self, prompt: str, *args, **kwargs) -> CompletionResponse:
        pass

    @abstractmethod
    def acompletion(self, prompt: str, *args, **kwargs) -> CompletionResponse:
        pass

    @abstractmethod
    def chat(
        self, message: List[ChatMessage], *args, **kwargs
    ) -> ChatCompletionResponse:
        pass

    @abstractmethod
    def achat(
        self, message: List[ChatMessage], *args, **kwargs
    ) -> ChatCompletionResponse:
        pass

    @abstractmethod
    def completion_stream(
        self, prompt: str, *args, **kwargs
    ) -> Generator[CompletionResponse]:
        raise ValueError("streaming is not supported")

    def acompletion_stream(
        self, prompt: str, *args, **kwargs
    ) -> Generator[CompletionResponse]:
        raise ValueError("streaming is not supported")

    def chat_stream(
        self, message: List[ChatMessage], *args, **kwargs
    ) -> Generator[ChatCompletionResponse]:
        raise ValueError("streaming is not supported")

    def achat_stream(
        self, message: List[ChatMessage], *args, **kwargs
    ) -> Generator[ChatCompletionResponse]:
        raise ValueError("streaming is not supported")
