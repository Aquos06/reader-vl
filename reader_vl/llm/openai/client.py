from typing import Optional

import numpy as np
from openai import OpenAI

from reader_vl.llm.client import llmBase
from reader_vl.llm.schemas import ChatCompletionResponse
from reader_vl.llm.utils import encode_image


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

    def _format_messages(self, prompt: str, image: Optional[np.ndarray]) -> list:
        """
        Formats the messages for OpenAI's API, including converting an image if provided.

        Args:
            prompt: The text prompt.
            image: A NumPy array representing an image (optional).

        Returns:
            A properly formatted list of messages.
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        if image is not None:
            base64_image = encode_image(image=image)
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        return messages

    def chat(
        self, prompt: str, image: Optional[np.ndarray], **kwargs
    ) -> ChatCompletionResponse:
        """
        Synchronously generates a chat completion for a list of chat messages using the OpenAI Chat Completions API.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Returns:
            A ChatCompletionResponse object containing the generated chat completion.
        """
        message = self._format_messages(prompt=prompt, image=image)
        response = self.client.chat.completions.create(
            model=self.model, messages=message, max_token=self.max_tokens, **kwargs
        )
        return ChatCompletionResponse(**response)

    def achat(
        self, prompt: str, image: Optional[np.ndarray], **kwargs
    ) -> ChatCompletionResponse:
        """
        Asynchronously generates a chat completion for a list of chat messages.
        This method currently calls the synchronous `chat` method.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Returns:
            A ChatCompletionResponse object containing the generated chat completion.
        """
        message = self._format_messages(prompt=prompt, image=image)
        return self.chat(message=message, **kwargs)
