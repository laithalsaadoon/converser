"""Models and their functionality using Converse."""

from typing import TypedDict


class ModelConfig(TypedDict):
    """A dictionary containing the results of the functionality checks."""

    model_name: str
    model_id: str
    converse: bool
    converse_stream: bool
    system_prompts: bool
    document_chat: bool
    vision: bool
    tool_use: bool
    streaming_tool_use: bool
    guardrails: bool


AMAZON_TITAN_TG1_LARGE: ModelConfig = {
    "model_name": "Titan Text Large",
    "model_id": "amazon.titan-tg1-large",
    "converse": True,
    "converse_stream": True,
    "system_prompts": False,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

AMAZON_TITAN_TEXT_LITE_V1: ModelConfig = {
    "model_name": "Titan Text G1 - Lite",
    "model_id": "amazon.titan-text-lite-v1",
    "converse": True,
    "converse_stream": True,
    "system_prompts": False,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

AMAZON_TITAN_TEXT_EXPRESS_V1: ModelConfig = {
    "model_name": "Titan Text G1 - Express",
    "model_id": "amazon.titan-text-express-v1",
    "converse": True,
    "converse_stream": True,
    "system_prompts": False,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

AMAZON_TITAN_TEXT_AGILE_V1: ModelConfig = {
    "model_name": "Titan Text G1 - Agile",
    "model_id": "amazon.titan-text-agile-v1",
    "converse": False,
    "converse_stream": False,
    "system_prompts": False,
    "document_chat": False,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": False,
}

AI21_J2_GRANDE_INSTRUCT: ModelConfig = {
    "model_name": "J2 Grande Instruct",
    "model_id": "ai21.j2-grande-instruct",
    "converse": True,
    "converse_stream": False,
    "system_prompts": False,
    "document_chat": False,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

AI21_J2_JUMBO_INSTRUCT: ModelConfig = {
    "model_name": "J2 Jumbo Instruct",
    "model_id": "ai21.j2-jumbo-instruct",
    "converse": True,
    "converse_stream": False,
    "system_prompts": False,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

AI21_J2_MID: ModelConfig = {
    "model_name": "Jurassic-2 Mid",
    "model_id": "ai21.j2-mid",
    "converse": True,
    "converse_stream": False,
    "system_prompts": False,
    "document_chat": False,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

AI21_J2_MID_V1: ModelConfig = {
    "model_name": "Jurassic-2 Mid",
    "model_id": "ai21.j2-mid-v1",
    "converse": True,
    "converse_stream": False,
    "system_prompts": False,
    "document_chat": False,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

AI21_J2_ULTRA: ModelConfig = {
    "model_name": "Jurassic-2 Ultra",
    "model_id": "ai21.j2-ultra",
    "converse": True,
    "converse_stream": False,
    "system_prompts": False,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

AI21_J2_ULTRA_V1: ModelConfig = {
    "model_name": "Jurassic-2 Ultra",
    "model_id": "ai21.j2-ultra-v1",
    "converse": True,
    "converse_stream": False,
    "system_prompts": False,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

ANTHROPIC_CLAUDE_INSTANT_V1: ModelConfig = {
    "model_name": "Claude Instant",
    "model_id": "anthropic.claude-instant-v1",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

ANTHROPIC_CLAUDE_V2_1: ModelConfig = {
    "model_name": "Claude",
    "model_id": "anthropic.claude-v2:1",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

ANTHROPIC_CLAUDE_V2: ModelConfig = {
    "model_name": "Claude",
    "model_id": "anthropic.claude-v2",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

ANTHROPIC_CLAUDE_3_SONNET_20240229_V1_0: ModelConfig = {
    "model_name": "Claude 3 Sonnet",
    "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": True,
    "tool_use": True,
    "streaming_tool_use": True,
    "guardrails": True,
}

ANTHROPIC_CLAUDE_3_HAIKU_20240307_V1_0: ModelConfig = {
    "model_name": "Claude 3 Haiku",
    "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": True,
    "tool_use": True,
    "streaming_tool_use": True,
    "guardrails": True,
}

ANTHROPIC_CLAUDE_3_OPUS_20240229_V1_0: ModelConfig = {
    "model_name": "Claude 3 Opus",
    "model_id": "anthropic.claude-3-opus-20240229-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": True,
    "tool_use": True,
    "streaming_tool_use": True,
    "guardrails": True,
}

COHERE_COMMAND_TEXT_V14: ModelConfig = {
    "model_name": "Command",
    "model_id": "cohere.command-text-v14",
    "converse": True,
    "converse_stream": True,
    "system_prompts": False,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

COHERE_COMMAND_R_V1_0: ModelConfig = {
    "model_name": "Command R",
    "model_id": "cohere.command-r-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": False,
    "tool_use": True,
    "streaming_tool_use": False,
    "guardrails": True,
}

COHERE_COMMAND_R_PLUS_V1_0: ModelConfig = {
    "model_name": "Command R+",
    "model_id": "cohere.command-r-plus-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": False,
    "tool_use": True,
    "streaming_tool_use": False,
    "guardrails": True,
}

COHERE_COMMAND_LIGHT_TEXT_V14: ModelConfig = {
    "model_name": "Command Light",
    "model_id": "cohere.command-light-text-v14",
    "converse": True,
    "converse_stream": True,
    "system_prompts": False,
    "document_chat": False,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

META_LLAMA3_8B_INSTRUCT_V1_0: ModelConfig = {
    "model_name": "Llama 3 8B Instruct",
    "model_id": "meta.llama3-8b-instruct-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

META_LLAMA3_70B_INSTRUCT_V1_0: ModelConfig = {
    "model_name": "Llama 3 70B Instruct",
    "model_id": "meta.llama3-70b-instruct-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

MISTRAL_MISTRAL_7B_INSTRUCT_V0_2: ModelConfig = {
    "model_name": "Mistral 7B Instruct",
    "model_id": "mistral.mistral-7b-instruct-v0:2",
    "converse": True,
    "converse_stream": True,
    "system_prompts": False,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

MISTRAL_MIXTRAL_8X7B_INSTRUCT_V0_1: ModelConfig = {
    "model_name": "Mixtral 8x7B Instruct",
    "model_id": "mistral.mixtral-8x7b-instruct-v0:1",
    "converse": True,
    "converse_stream": True,
    "system_prompts": False,
    "document_chat": True,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

MISTRAL_MISTRAL_LARGE_2402_V1_0: ModelConfig = {
    "model_name": "Mistral Large",
    "model_id": "mistral.mistral-large-2402-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": False,
    "tool_use": True,
    "streaming_tool_use": False,
    "guardrails": True,
}

AMAZON_TITAN_TEXT_PREMIER_V1_0: ModelConfig = {
    "model_name": "Titan Text G1 - Premier",
    "model_id": "amazon.titan-text-premier-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": False,
    "document_chat": False,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

AI21_JAMBA_INSTRUCT_V1_0: ModelConfig = {
    "model_name": "Jamba-Instruct",
    "model_id": "ai21.jamba-instruct-v1:0",
    "converse": True,
    "converse_stream": False,
    "system_prompts": True,
    "document_chat": False,
    "vision": False,
    "tool_use": False,
    "streaming_tool_use": False,
    "guardrails": True,
}

ANTHROPIC_CLAUDE_3_5_SONNET_20240620_V1_0: ModelConfig = {
    "model_name": "Claude 3.5 Sonnet",
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": True,
    "vision": True,
    "tool_use": True,
    "streaming_tool_use": True,
    "guardrails": True,
}

MISTRAL_MISTRAL_SMALL_2402_V1_0: ModelConfig = {
    "model_name": "Mistral Small",
    "model_id": "mistral.mistral-small-2402-v1:0",
    "converse": True,
    "converse_stream": True,
    "system_prompts": True,
    "document_chat": False,
    "vision": False,
    "tool_use": True,
    "streaming_tool_use": False,
    "guardrails": True,
}
