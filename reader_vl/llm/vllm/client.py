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
    def __init__(
        self, url: str, model: str, temperature: float, max_tokens: Optional[int]
    ) -> None:
        super().__init__(url=url, model=model, max_tokens=max_tokens)
        self.temperature = temperature
        self.completion_url = f"{url}/v1/completions"
        self.chat_url = f"{url}/v1/chat/completions"

    async def _aprocess_stream_chunk(self, chunk: str, response_type):
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
        return {
            "temperature": self.temperature,
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

    def get_chat_params(self, message: List[ChatMessage], **kwargs) -> dict:
        return {
            "temperature": self.temperature,
            "model": self.model,
            "messages": message,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

    def completion(self, prompt, **kwargs) -> CompletionResponse:
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
        params = self.get_chat_params(message=message, **kwargs)
        params["stream"] = True

        response = requests.post(self.chat_url, json=params, stream=True)
        response.raise_for_status()
        for chunk in response.iter_lines(decode_unicode="utf-8"):
            processed_chunk = self._process_stream_chunk(chunk, ChatCompletionResponse)
            yield processed_chunk
