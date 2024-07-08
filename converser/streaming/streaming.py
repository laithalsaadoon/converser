"""This module contains the functions that are used to interact with the Bedrock Runtime API."""

from converser.conversation_memory.memory import Memory
from converser.models.models import ConverseStreamingKeys, InferenceConfig
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime.type_defs import (
    ConverseStreamOutputTypeDef,
    ConverseStreamResponseTypeDef,
    InferenceConfigurationTypeDef,
    MessageUnionTypeDef,
    SystemContentBlockTypeDef,
)
from typing import Any, Generator, List, Optional, Sequence, cast


class ConverserStreamOutputTypeDefEnd(ConverseStreamOutputTypeDef):
    """Extend the ConverseStreamOutputTypeDef with the 'done' key."""

    done: bool


def stream_messages(
    client: BedrockRuntimeClient,
    model_id: str,
    messages: List[MessageUnionTypeDef],
    system_prompt: Sequence[SystemContentBlockTypeDef],
    memory: Optional[Memory] = None,
    inference_config: InferenceConfig = InferenceConfig(),
    stdout: Optional[bool] = None,
) -> Generator[tuple[ConverserStreamOutputTypeDefEnd, None | MessageUnionTypeDef], Any, Any]:
    """Stream messages to the model."""
    response: ConverseStreamResponseTypeDef = client.converse_stream(
        modelId=model_id,
        messages=messages,
        system=system_prompt,
        inferenceConfig=cast(InferenceConfigurationTypeDef, inference_config.model_dump()),
    )

    complete_message: list[str] = []
    for event in response['stream']:
        yield_message: ConverserStreamOutputTypeDefEnd = cast(
            ConverserStreamOutputTypeDefEnd, {**event, **{'done': False}}
        )
        # check which event type is in the response and assign the correct output key
        output_key = next((key for key in event.keys() if key in ConverseStreamingKeys), None)
        match output_key:
            case (
                ConverseStreamingKeys.MESSAGE_START
                | ConverseStreamingKeys.CONTENT_BLOCK_START
                | ConverseStreamingKeys.CONTENT_BLOCK_STOP
                | ConverseStreamingKeys.METADATA
            ):
                yield yield_message, None
            case ConverseStreamingKeys.CONTENT_BLOCK_DELTA:
                text = event['contentBlockDelta']['delta']['text']  # type: ignore
                print(text, end='') if stdout else None
                complete_message.append(text)
                yield yield_message, None
            case ConverseStreamingKeys.MESSAGE_STOP:
                final_text = ''.join(complete_message)
                final_message: MessageUnionTypeDef = {
                    'role': 'assistant',
                    'content': [{'text': final_text}],
                }
                if memory:
                    memory.add_messages([messages[-1]] + [final_message])
                yield_message['done'] = True
                yield yield_message, final_message
                complete_message = []
            case None:
                raise ValueError('Invalid event type')
