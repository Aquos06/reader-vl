from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from reader_vl.llm.schemas import (
    ChatCompletionResponse,
    ChatContent,
    ChatMessage,
    ChatRole,
    ContentType,
)
from reader_vl.llm.utils import encode_image


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

    def _format_messages(
        self, prompt: str, image: np.ndarray, return_dict: bool = True
    ) -> list:
        """
        Formats the messages for OpenAI's API, including converting an image if provided.

        Args:
            prompt: The text prompt.
            image: A NumPy array representing an image (optional).

        Returns:
            A properly formatted list of messages.
        """
        base64_image = encode_image(image=image)
        messages = [
            ChatMessage(
                role=ChatRole.USER.value,
                content=[
                    ChatContent(type=ContentType.TEXT.value, text=prompt),
                    ChatContent(
                        type=ContentType.IMAGE.value,
                        image_url={"url": f"data:image/png;base64,{base64_image}"},
                    ),
                ],
            )
        ]

        if return_dict:
            return [message.model_dump(exclude_none=True) for message in messages]

        return messages

    @abstractmethod
    def chat(
        self, prompt: str, image: np.ndarray, *args, **kwargs
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
        self, prompt: str, image: np.ndarray, *args, **kwargs
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
