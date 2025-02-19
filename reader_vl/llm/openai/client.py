from typing import List, Optional

from openai import OpenAI

from reader_vl.llm.client import llmBase
from reader_vl.llm.schemas import (
    ChatCompletionResponse,
    ChatMessage,
    CompletionResponse,
)


class OpenAIClient(llmBase):
    def __init__(self, api_key: str, model: str, max_tokens: Optional[int] = None):
        self.client = OpenAI(api_key=api_key)
        super().__init__(url="", model=model, max_tokens=max_tokens)

    def completion(self, prompt, **kwargs) -> CompletionResponse:
        response = self.client.completions.create(
            model=self.model, prompt=prompt, max_tokens=self.max_tokens, **kwargs
        )
        return CompletionResponse(**response)

    def acompletion(self, prompt, **kwargs) -> CompletionResponse:
        return self.completion(prompt=prompt, **kwargs)

    def chat(self, message: List[ChatMessage], **kwargs) -> ChatCompletionResponse:
        response = self.client.chat.completions.create(
            model=self.model, messages=message, max_token=self.max_tokens, **kwargs
        )
        return ChatCompletionResponse(**response)

    def achat(self, message: List[ChatMessage], **kwargs) -> ChatCompletionResponse:
        return self.chat(message=message, **kwargs)
