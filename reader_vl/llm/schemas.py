from typing import List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

class ChoiceBase(BaseModel):
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None 

class ResponseBase(BaseModel):
    created: int
    model: str

class CompletionChoice(ChoiceBase):
    text: str
    index: int


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(ResponseBase):
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ChatMessage(BaseModel):
    role: str
    content: str


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
    
class ChatRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"