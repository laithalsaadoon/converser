"""Models for the converser API."""

from enum import Enum
from pydantic import BaseModel, Field


class ConverseStreamingKeys(str, Enum):
    """Keys for the streaming response."""

    CONTENT_BLOCK_DELTA = 'contentBlockDelta'
    MESSAGE_START = 'messageStart'
    MESSAGE_STOP = 'messageStop'
    CONTENT_BLOCK_START = 'contentBlockStart'
    CONTENT_BLOCK_STOP = 'contentBlockStop'
    METADATA = 'metadata'


# Inference Configuration Class
class InferenceConfig(BaseModel):
    """A class to store inference configuration."""

    temperature: float = Field(default=1, ge=0, le=1, description='Temperature for LLM')
    maxTokens: int = Field(default=4096, ge=1, le=4096, description='Max tokens for LLM')
    topP: float = Field(default=0.999, ge=0, le=1, description='Top p for LLM')
    stopSequences: list[str] = Field(
        default=[], description='Stop sequences for LLM', max_length=4
    )
