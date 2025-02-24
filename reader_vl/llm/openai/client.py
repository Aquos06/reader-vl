from typing import Iterator, List, Optional

from openai import OpenAI

from reader_vl.llm.client import llmBase
from reader_vl.llm.schemas import (
    ChatCompletionResponse,
    ChatMessage,
    CompletionResponse,
)


class OpenAIClient(llmBase):
    """
    Client for interacting with the OpenAI API.
    Inherits from llmBase and implements its abstract methods.
    """

    def __init__(self, api_key: str, model: str, max_tokens: Optional[int] = None):
        """
        Initializes the OpenAIClient object.

        Args:
            api_key: The API key for accessing the OpenAI API.
            model: The name of the OpenAI model to use.
            max_tokens: The maximum number of tokens to generate (optional).
        """
        self.client = OpenAI(api_key=api_key)
        super().__init__(url="", model=model, max_tokens=max_tokens)

    def completion(self, prompt, **kwargs) -> CompletionResponse:
        """
        Synchronously generates a completion for a given prompt using the OpenAI Completions API.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the completion request.

        Returns:
            A CompletionResponse object containing the generated completion.
        """
        response = self.client.completions.create(
            model=self.model, prompt=prompt, max_tokens=self.max_tokens, **kwargs
        )
        return CompletionResponse(**response)

    def acompletion(self, prompt, **kwargs) -> CompletionResponse:
        """
        Asynchronously generates a completion for a given prompt.
        This method currently calls the synchronous `completion` method.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the completion request.

        Returns:
            A CompletionResponse object containing the generated completion.
        """
        return self.completion(prompt=prompt, **kwargs)

    def chat(self, message: List[ChatMessage], **kwargs) -> ChatCompletionResponse:
        """
        Synchronously generates a chat completion for a list of chat messages using the OpenAI Chat Completions API.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Returns:
            A ChatCompletionResponse object containing the generated chat completion.
        """
        response = self.client.chat.completions.create(
            model=self.model, messages=message, max_token=self.max_tokens, **kwargs
        )
        return ChatCompletionResponse(**response)

    def achat(self, message: List[ChatMessage], **kwargs) -> ChatCompletionResponse:
        """
        Asynchronously generates a chat completion for a list of chat messages.
        This method currently calls the synchronous `chat` method.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Returns:
            A ChatCompletionResponse object containing the generated chat completion.
        """
        return self.chat(message=message, **kwargs)

    def completion_stream(self, prompt, **kwargs) -> Iterator[CompletionResponse]:
        """
        Synchronously generates a completion stream for a given prompt using the OpenAI Completions API with streaming enabled.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the completion request.

        Yields:
            An Iterator of CompletionResponse objects, each representing a chunk of the completion.
        """
        stream = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield CompletionResponse(**chunk)

    def acompletion_stream(self, prompt, **kwargs) -> Iterator[CompletionResponse]:
        """
        Asynchronously generates a completion stream for a given prompt.
        This method currently calls the synchronous `completion_stream` method.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the completion request.

        Returns:
            An Iterator of CompletionResponse objects, each representing a chunk of the completion.
        """
        return self.completion_stream(prompt=prompt, **kwargs)

    def chat_stream(
        self, message: List[ChatMessage], **kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Synchronously generates a chat completion stream for a list of chat messages using the OpenAI Chat Completions API with streaming enabled.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Yields:
            An Iterator of ChatCompletionResponse objects, each representing a chunk of the completion.
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            yield ChatCompletionResponse(**chunk)

    def achat_stream(
        self, message: List[ChatMessage], **kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Asynchronously generates a chat completion stream for a list of chat messages.
        This method currently calls the synchronous `chat_stream` method.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Returns:
            An Iterator of ChatCompletionResponse objects, each representing a chunk of the completion.
        """
        return self.chat_stream(message=message, **kwargs)
