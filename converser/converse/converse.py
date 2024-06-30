"""This module contains the Converse class."""

from converser.conversation_memory import Memory
from converser.models import InferenceConfig, ModelId
from converser.streaming import stream_messages
from converser.utils import get_bedrock_client
from converser.utils.helpers import sanitize_file_name
from functools import partial
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime.literals import (
    DocumentFormatType,
    ImageFormatType,
)
from mypy_boto3_bedrock_runtime.type_defs import (
    ContentBlockTypeDef,
    ConverseResponseTypeDef,
    InferenceConfigurationTypeDef,
    MessageOutputTypeDef,
    MessageTypeDef,
    SystemContentBlockTypeDef,
)
from pathlib import Path
from typing import (
    List,
    Literal,
    Optional,
    Sequence,
    cast,
    get_args,
)


class Converse:
    """The Converse class is used to interact with the Bedrock Runtime API."""

    def __init__(
        self,
        model_id: ModelId,
        system_prompt: Optional[SystemContentBlockTypeDef] = None,
        memory: Optional[Memory] = None,
        inference_config: InferenceConfig = InferenceConfig(),
        region: str = 'us-west-2',
    ):
        """Initialize the Converse class.

        Args:
            model_id (ModelId): The ID of the model to use for conversation.
            system_prompt (Optional[SystemContentBlockTypeDef], optional): The system prompt to use. Defaults to None.
            memory (Optional[Memory], optional): The memory object to use for conversation. Defaults to None.
            inference_config (InferenceConfig, optional): The inference configuration to use. Defaults to InferenceConfig().
            region (str, optional): The region to use for the Bedrock client. Defaults to 'us-west-2'.
        """  # noqa: E501
        self.client: BedrockRuntimeClient = get_bedrock_client(region=region)
        self.model_id = model_id
        self.system_prompt: Sequence[SystemContentBlockTypeDef] = (
            [system_prompt] if system_prompt else []
        )
        self.memory = memory
        self.inference_config = inference_config
        self.stream_messages = partial(stream_messages, **self.__dict__)

    def send_messages(self, messages: List[MessageTypeDef]) -> ConverseResponseTypeDef:
        """Send a message to the model.

        Args:
            messages (MessageTypeDef): The messages to send.

        Returns:
            ConverseResponseTypeDef: The response from the model.
        """
        if self.memory:
            self.memory.add_messages(messages)
            messages = self.memory.get_history()
        else:
            messages = messages

        response: ConverseResponseTypeDef = self.client.converse(
            modelId=self.model_id.value,
            messages=messages,
            system=self.system_prompt,
            inferenceConfig=cast(
                InferenceConfigurationTypeDef, self.inference_config.model_dump()
            ),
        )

        match response['stopReason']:
            case 'end_turn' | 'tool_use' | 'max_tokens' | 'stop_sequence':
                content = response['output']['message']['content']  # type: ignore - the keys are checked in the match statement
                if self.memory:
                    assistant_message: MessageOutputTypeDef = {
                        'role': 'assistant',
                        'content': content,
                    }
                    self.memory.add_messages([assistant_message])
            # default case
            case _:
                raise NotImplementedError(
                    f"Stop reason '{response['stopReason']}' not implemented"
                )

        return response

    def from_file(
        self,
        file_path: str,
        content_type: Literal['image', 'document'],
        user_text: str = 'Please describe the contents of the file in detail',
    ) -> ConverseResponseTypeDef:
        """Create a message from a file.

        Args:
            file_path (str): The path to the file.
            content_type (Literal['image', 'document']): The type of content in the file.
            user_text (Optional[str], optional): The user text to include with the file. Defaults to None.

        Returns:
            ConverseResponseTypeDef: The response from the model.

        Raises:
            ValueError: If the document format or image format is unsupported.
        """  # noqa: E501
        with open(file_path, 'rb') as file:
            content_bytes = file.read()

        content_block: ContentBlockTypeDef

        # check the document extension and make sure it matches DocumentFormatType
        file_extension = Path(file_path).suffix[1:]
        if content_type == 'document' and file_extension not in get_args(DocumentFormatType):
            # handle invalid document format
            raise ValueError(f'Invalid document format for file: {file_path}')
        if content_type == 'image' and file_extension not in get_args(ImageFormatType):
            raise ValueError(f'Unsupported image format: {file_extension}')

        document_name: str = sanitize_file_name(file_path)

        if content_type == 'document':
            content_block = {
                'document': {
                    'format': file_extension,  # type: ignore
                    'name': document_name,
                    'source': {'bytes': content_bytes},
                }
            }
        elif content_type == 'image':
            content_block = {
                'image': {
                    'format': file_extension,  # type: ignore
                    'source': {'bytes': content_bytes},
                }
            }
        else:
            raise ValueError('Unsupported content type')

        user_message: MessageTypeDef = {
            'role': 'user',
            'content': [{'text': user_text}, content_block],
        }
        return self.send_messages([user_message])
