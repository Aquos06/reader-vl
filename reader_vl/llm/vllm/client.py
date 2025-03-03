import json
import logging
from typing import List, Optional

import httpx
import numpy as np
import requests

from reader_vl.llm.client import llmBase
from reader_vl.llm.schemas import ChatCompletionResponse

logging.basicConfig(level=logging.INFO)


class VllmClient(llmBase):
    """
    Client for interacting with the VLLM API.
    Inherits from llmBase and implements its abstract methods.
    """

    def __init__(
        self, url: str, model: str, temperature: float, max_tokens: Optional[int]
    ) -> None:
        """
        Initializes the VllmClient object.

        Args:
            url: The base URL of the VLLM API.
            model: The name of the VLLM model to use.
            temperature: The temperature parameter for controlling randomness.
            max_tokens: The maximum number of tokens to generate (optional).
        """
        super().__init__(url=url, model=model, max_tokens=max_tokens)
        self.temperature = temperature
        self.completion_url = f"{url}/v1/completions"
        self.chat_url = f"{url}/v1/chat/completions"

    def get_chat_params(self, messages: List[dict], **kwargs) -> dict:
        """
        Constructs the parameters for the chat completion request.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Returns:
            A dictionary containing the parameters for the chat completion request.
        """
        return {
            "temperature": self.temperature,
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

    def chat(self, prompt: str, image: np.ndarray, **kwargs) -> ChatCompletionResponse:
        """
        Synchronously generates a chat completion for a list of chat messages.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Returns:
            A ChatCompletionResponse object containing the generated chat completion.

        Raises:
            ValueError: If there is a JSON decode error or any other error during the request.
        """
        messages = self._format_messages(prompt=prompt, image=image)
        response = requests.post(
            url=self.chat_url, json=self.get_chat_params(messages=messages, **kwargs)
        )
        try:
            response.raise_for_status()
            response = response.json()
            return ChatCompletionResponse(**response)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}, with response: {response}")
            raise ValueError(f"Json decode error in chat completion: {e}")
        except Exception as e:
            logging.error(e)
            raise ValueError(f"An unexpected error occured during chat completion: {e}")

    async def achat(
        self, prompt: str, image: np.ndarray, **kwargs
    ) -> ChatCompletionResponse:
        """
        Asynchronously generates a chat completion for a list of chat messages.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Returns:
            A ChatCompletionResponse object containing the generated chat completion.

        Raises:
            ValueError: If there is a JSON decode error or any other error during the request.
        """
        messages = self._format_messages(prompt=prompt, image=image)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url=self.chat_url,
                    json=self.get_chat_params(messages=messages, **kwargs),
                )
                response.raise_for_status()
                response = response.json()
                return ChatCompletionResponse(**response)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}, with response: {response}")
                raise ValueError(f"Json decode error in chat completion: {e}")
            except Exception as e:
                logging.error(e)
                raise ValueError(
                    f"An unexpected error occured during chat completion: {e}"
                )
