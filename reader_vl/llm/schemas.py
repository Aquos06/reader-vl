from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel


class ChatRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image_url"


class ChoiceBase(BaseModel):
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class ResponseBase(BaseModel):
    created: int
    model: str


class ChatContent(BaseModel):
    type: ContentType
    text: Optional[str] = None
    image_url: Optional[dict] = None


class ChatMessage(BaseModel):
    role: ChatRole
    content: List[ChatContent]


class ChatCompletionChoice(ChoiceBase):
    message: ChatMessage
    index: int


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(ResponseBase):
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
