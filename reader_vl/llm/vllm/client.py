import json
import logging
import re
from typing import AsyncGenerator, Generator, List, Optional

import httpx
import requests

from reader_vl.llm.client import llmBase
from reader_vl.llm.schemas import (
    ChatCompletionResponse,
    ChatMessage,
    CompletionResponse,
)

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

    async def _aprocess_stream_chunk(self, chunk: str, response_type):
        """
        Asynchronously processes a single chunk from the streaming response.

        Args:
            chunk: The chunk of data received from the stream.
            response_type: The type of response object to create (CompletionResponse or ChatCompletionResponse).

        Returns:
            A response object (CompletionResponse or ChatCompletionResponse) created from the chunk.

        Raises:
            ValueError: If the chunk has an invalid format or JSON decode error.
        """
        match = re.match(r"^data: ?", chunk)
        if match:
            json_string = chunk[match.end() :]
            try:
                json_chunk = json.loads(json_string)
                return response_type(**json_chunk)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}, chunk: {chunk}")
                return ValueError(f"Invalid JSON response: {e}")
            except Exception as e:
                logging.error(f"Unexpected chunk format: {chunk}, with error: {e}")
                raise ValueError(
                    f"An unxepected error occured during streaming, with error :{e}"
                )

    def _process_stream_chunk(self, chunk: str, response_type):
        """
        Processes a single chunk from the streaming response.

        Args:
            chunk: The chunk of data received from the stream.
            response_type: The type of response object to create (CompletionResponse or ChatCompletionResponse).

        Returns:
            A response object (CompletionResponse or ChatCompletionResponse) created from the chunk.

        Raises:
            ValueError: If the chunk has an invalid format or JSON decode error.
        """
        match = re.match(r"^data: ?", chunk)
        if match:
            json_string = chunk[match.end() :]
            try:
                json_chunk = json.loads(json_string)
                return response_type(**json_chunk)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}, chunk: {chunk}")
                raise ValueError(f"JSON decode error: {e}")
            except Exception as e:
                logging.error(f"Unexpected chunk format: {chunk}, with error: {e}")
                raise ValueError(
                    f"An unxepected error occured during the streaming, with error: {e}"
                )

    def get_completion_params(self, prompt: str, **kwargs) -> dict:
        """
        Constructs the parameters for the completion request.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the completion request.

        Returns:
            A dictionary containing the parameters for the completion request.
        """
        return {
            "temperature": self.temperature,
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

    def get_chat_params(self, message: List[ChatMessage], **kwargs) -> dict:
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
            "messages": message,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

    def completion(self, prompt, **kwargs) -> CompletionResponse:
        """
        Synchronously generates a completion for a given prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the completion request.

        Returns:
            A CompletionResponse object containing the generated completion.

        Raises:
            ValueError: If there is a JSON decode error or any other error during the request.
        """
        response = requests.post(
            url=self.completion_url,
            json=self.get_completion_params(prompt=prompt, **kwargs),
        )
        try:
            response.raise_for_status()
            response = response.json()
            return CompletionResponse(**response)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}, with response: {response}")
            raise ValueError(f"Json decode error in completion: {e}")
        except Exception as e:
            logging.error(e)
            raise ValueError(f"An unexpected error occured during completion: {e}")

    async def acompletion(self, prompt, **kwargs) -> CompletionResponse:
        """
        Asynchronously generates a completion for a given prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the completion request.

        Returns:
            A CompletionResponse object containing the generated completion.

        Raises:
            ValueError: If there is a JSON decode error or any other error during the request.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url=self.completion_url,
                    json=self.get_completion_params(prompt=prompt, **kwargs),
                )
                response.raise_for_status()
                response = response.json()
                return CompletionResponse(**response)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}, with response: {response}")
                raise ValueError(f"Json decode error in completion: {e}")
            except Exception as e:
                logging.error(e)
                raise ValueError(f"An unexpected error occured during completion: {e}")

    def chat(self, message, **kwargs) -> ChatCompletionResponse:
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
        response = requests.post(
            url=self.chat_url, json=self.get_chat_params(message=message, **kwargs)
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

    async def achat(self, message, **kwargs) -> ChatCompletionResponse:
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
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url=self.chat_url,
                    json=self.get_chat_params(message=message, **kwargs),
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

    def completion_stream(self, prompt: str, **kwargs) -> Generator[CompletionResponse]:
        """
        Synchronously generates a completion stream for a given prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the completion request.

        Yields:
            A Generator of CompletionResponse objects, each representing a chunk of the completion.
        """
        params = self.get_completion_params(prompt=prompt, **kwargs)
        params["stream"] = True

        response = requests.post(self.completion_url, json=params, stream=True)
        response.raise_for_status()
        for chunk in response.iter_lines(decode_unicode="utf-8"):
            processed_chunk = self._process_stream_chunk(chunk, CompletionResponse)
            yield processed_chunk

    async def acompletion_stream(
        self, prompt: str, **kwargs
    ) -> AsyncGenerator[CompletionResponse]:
        """
        Asynchronously generates a completion stream for a given prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the completion request.

        Yields:
            An AsyncGenerator of CompletionResponse objects, each representing a chunk of the completion.

        Raises:
            ValueError: If there is a JSON decode error or any other error during the request.
        """
        params = self.get_completion_params(prompt=prompt, **kwargs)
        params["stream"] = True

        try:
            async with httpx.AsyncClient as client:
                async with client.post(
                    self.completion_url, json=params, stream=True
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_lines():
                        if chunk:
                            processed_chunk = await self._aprocess_stream_chunk(
                                chunk, CompletionResponse
                            )
                            yield processed_chunk

        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}, with response: {response}")
            raise ValueError(f"Json decode error in completion: {e}")
        except Exception as e:
            logging.error(e)
            raise ValueError(f"An unexpected error occured during completion: {e}")

    async def achat_stream(
        self, message: List[ChatMessage], **kwargs
    ) -> AsyncGenerator[ChatCompletionResponse]:
        """
        Asynchronously generates a chat completion stream for a list of chat messages.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Yields:
            An AsyncGenerator of ChatCompletionResponse objects, each representing a chunk of the completion.

        Raises:
            ValueError: If there is a JSON decode error or any other error during the request.
        """
        params = self.get_chat_params(message=message, **kwargs)
        params["stream"] = True

        try:
            async with httpx.AsyncClient as client:
                async with client.post(
                    self.chat_url, json=params, stream=True
                ) as response:
                    response.raise_for_status()

                    async for chunk in response.aiter_lines():
                        if chunk:
                            processed_chunk = await self._aprocess_stream_chunk(
                                chunk, ChatCompletionResponse
                            )
                            yield processed_chunk

        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}, with response: {response}")
            raise ValueError(f"Json decode error in chat completion: {e}")
        except Exception as e:
            logging.error(e)
            raise ValueError(f"An unexpected error occured during chat completion: {e}")

    def chat_stream(
        self, message: List[ChatMessage], **kwargs
    ) -> Generator[ChatCompletionResponse]:
        """
        Synchronously generates a chat completion stream for a list of chat messages.

        Args:
            message: A list of ChatMessage objects representing the conversation history.
            **kwargs: Additional parameters for the chat completion request.

        Yields:
            A Generator of ChatCompletionResponse objects, each representing a chunk of the completion.
        """
        params = self.get_chat_params(message=message, **kwargs)
        params["stream"] = True

        response = requests.post(self.chat_url, json=params, stream=True)
        response.raise_for_status()
        for chunk in response.iter_lines(decode_unicode="utf-8"):
            processed_chunk = self._process_stream_chunk(chunk, ChatCompletionResponse)
            yield processed_chunk
