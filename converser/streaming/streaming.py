"""This module contains the functions that are used to interact with the Bedrock Runtime API."""

from converser.conversation_memory.memory import Memory
from converser.models.models import ConverseStreamingKeys, InferenceConfig, ModelId
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime.type_defs import (
    ConverseStreamResponseTypeDef,
    InferenceConfigurationTypeDef,
    MessageTypeDef,
    SystemContentBlockTypeDef,
)
from typing import Any, Dict, Generator, List, Optional, cast


def stream_messages(
    client: BedrockRuntimeClient,
    model_id: ModelId,
    messages: List[MessageTypeDef],
    system_prompt: Optional[SystemContentBlockTypeDef] = None,
    memory: Optional[Memory] = None,
    inference_config: InferenceConfig = InferenceConfig(),
    stdout: Optional[bool] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Stream messages to the model."""
    if memory:
        memory.add_messages(messages)
        messages = memory.get_history()
    else:
        messages = messages

    response: ConverseStreamResponseTypeDef = client.converse_stream(
        modelId=model_id.value,
        messages=messages,
        system=[system_prompt] if system_prompt else [],
        inferenceConfig=cast(InferenceConfigurationTypeDef, inference_config.model_dump()),
    )

    complete_message: list[str] = []
    for event in response['stream']:
        # check which event type is in the response and assign the correct output key
        output_key = next((key for key in event.keys() if key in ConverseStreamingKeys), None)
        match output_key:
            case (
                ConverseStreamingKeys.MESSAGE_START
                | ConverseStreamingKeys.CONTENT_BLOCK_START
                | ConverseStreamingKeys.CONTENT_BLOCK_STOP
                | ConverseStreamingKeys.METADATA
            ):
                pass
            case ConverseStreamingKeys.CONTENT_BLOCK_DELTA:
                text = event['contentBlockDelta']['delta']['text']  # type: ignore
                print(text, end='') if stdout else None
                complete_message.append(text)
                yield {'role': 'assistant', 'content': [{'text': text}]}
            case ConverseStreamingKeys.MESSAGE_STOP:
                final_message = ''.join(complete_message)
                yield {
                    'role': 'assistant',
                    'content': [{'text': final_message}],
                    'stopReason': event['messageStop']['stopReason'],  # type: ignore
                }
                complete_message = []
            case None:
                raise ValueError('Invalid event type')

    if memory:
        memory.add_messages([{'role': 'assistant', 'content': complete_message}])
        messages = memory.get_history()
    else:
        messages = messages
