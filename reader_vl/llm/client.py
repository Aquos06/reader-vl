from abc import ABC, abstractmethod
from typing import Generator, List, Optional

from llm.schemas import ChatCompletionResponse, ChatMessage, CompletionResponse


class llmBase(ABC):
    """
    Abstract base class for LLM (Language Model) clients.
    Defines the interface for interacting with various LLM APIs.
    """

    def __init__(self, url: str, model: str, max_tokens: Optional[int]) -> None:
        """
        Initializes the llmBase object.

        Args:
            url: The URL of the LLM API.
            model: The name or identifier of the LLM model.
            max_tokens: The maximum number of tokens to generate (optional).
        """
        self.url = url
        self.model = model
        self.max_tokens = max_tokens

    @abstractmethod
    def completion(self, prompt: str, *args, **kwargs) -> CompletionResponse:
        """
        Synchronously generates a completion for a given prompt.

        Args:
            prompt: The input prompt string.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A CompletionResponse object containing the generated completion.
        """

    @abstractmethod
    def acompletion(self, prompt: str, *args, **kwargs) -> CompletionResponse:
        """
        Asynchronously generates a completion for a given prompt.

        Args:
            prompt: The input prompt string.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A CompletionResponse object containing the generated completion.
        """

    @abstractmethod
    def chat(
        self, message: List[ChatMessage], *args, **kwargs
    ) -> ChatCompletionResponse:
        """
        Synchronously generates a chat completion for a list of chat messages.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A ChatCompletionResponse object containing the generated chat completion.
        """

    @abstractmethod
    def achat(
        self, message: List[ChatMessage], *args, **kwargs
    ) -> ChatCompletionResponse:
        """
        Asynchronously generates a chat completion for a list of chat messages.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            A ChatCompletionResponse object containing the generated chat completion.
        """

    @abstractmethod
    def completion_stream(
        self, prompt: str, *args, **kwargs
    ) -> Generator[CompletionResponse]:
        """
        Synchronously generates a completion stream for a given prompt.

        Args:
            prompt: The input prompt string.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Yields:
            A Generator of CompletionResponse objects, each representing a chunk of the completion.

        Raises:
            ValueError: If streaming is not supported by the implementation.
        """
        raise ValueError("streaming is not supported")

    def acompletion_stream(
        self, prompt: str, *args, **kwargs
    ) -> Generator[CompletionResponse]:
        """
        Asynchronously generates a completion stream for a given prompt.

        Args:
            prompt: The input prompt string.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Yields:
            A Generator of CompletionResponse objects, each representing a chunk of the completion.

        Raises:
            ValueError: If streaming is not supported by the implementation.
        """
        raise ValueError("streaming is not supported")

    def chat_stream(
        self, message: List[ChatMessage], *args, **kwargs
    ) -> Generator[ChatCompletionResponse]:
        """
        Synchronously generates a chat completion stream for a list of chat messages.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Yields:
            A Generator of ChatCompletionResponse objects, each representing a chunk of the completion.

        Raises:
            ValueError: If streaming is not supported by the implementation.
        """
        raise ValueError("streaming is not supported")

    def achat_stream(
        self, message: List[ChatMessage], *args, **kwargs
    ) -> Generator[ChatCompletionResponse]:
        """
        Asynchronously generates a chat completion stream for a list of chat messages.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Yields:
            A Generator of ChatCompletionResponse objects, each representing a chunk of the completion.

        Raises:
            ValueError: If streaming is not supported by the implementation.
        """
        raise ValueError("streaming is not supported")
