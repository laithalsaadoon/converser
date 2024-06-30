from enum import Enum
from pydantic import BaseModel, Field


class ModelId(str, Enum):
    """Model ID enum so I don't have to remember the strings."""

    CLAUDE_3_HAIKU = 'anthropic.claude-3-haiku-20240307-v1:0'
    CLAUDE_3_SONNET = 'anthropic.claude-3-sonnet-20240229-v1:0'
    CLAUDE_3_OPUS = 'anthropic.claude-3-opus-20240229-v1:0'
    CLAUDE_3_5_SONNET = 'anthropic.claude-3-5-sonnet-20240620-v1:0'


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
