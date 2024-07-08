"""This module contains the Converse class."""

from converser.conversation_memory import Memory
from converser.models import InferenceConfig
from converser.streaming import ConverserStreamOutputTypeDefEnd, stream_messages
from converser.utils import get_bedrock_client
from converser.utils.helpers import sanitize_file_name
from functools import partial, wraps
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime.literals import (
    DocumentFormatType,
    ImageFormatType,
)
from mypy_boto3_bedrock_runtime.type_defs import (
    ContentBlockTypeDef,
    ConverseResponseTypeDef,
    InferenceConfigurationTypeDef,
    MessageTypeDef,
    MessageUnionTypeDef,
    SystemContentBlockTypeDef,
)
from pathlib import Path
from typing import (
    Any,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
    get_args,
    overload,
)


def validate_message_order(func):
    """Decorator to validate the message order."""

    @wraps(func)
    def wrapper(
        self, messages: List[MessageUnionTypeDef], *args, **kwargs
    ) -> Union[
        ConverseResponseTypeDef,
        Generator[tuple[ConverserStreamOutputTypeDefEnd, None | MessageUnionTypeDef], Any, Any],
    ]:
        if not self._is_valid_message_order(messages):
            raise ValueError(
                'Invalid message order: Messages must start with a user message '
                'and alternate between user and assistant.'
            )
        return func(self, messages, *args, **kwargs)

    return wrapper


class Converse:
    """The Converse class is used to interact with the Bedrock Runtime API."""

    def __init__(
        self,
        model_id: str,
        system_prompt: Optional[SystemContentBlockTypeDef] = None,
        memory: Optional[Memory] = None,
        inference_config: InferenceConfig = InferenceConfig(),
        region: str = 'us-west-2',
        client: Optional[BedrockRuntimeClient] = None,
    ):
        """Initialize the Converse class.

        Args:
            model_id (ModelId): The ID of the model to use for conversation.
            system_prompt (Optional[SystemContentBlockTypeDef], optional): The system prompt to use. Defaults to None.
            memory (Optional[Memory], optional): The memory object to use for conversation. Defaults to None.
            inference_config (InferenceConfig, optional): The inference configuration to use. Defaults to InferenceConfig().
            region (str, optional): The region to use for the Bedrock client. Defaults to 'us-west-2'.
            client (Optional[BedrockRuntimeClient], optional): The Bedrock client to use. It's best if you pass your own client, but one will be created if you don't. Defaults to None.
        """  # noqa: E501
        self.client = get_bedrock_client(region=region) if client is None else client
        self.model_id = model_id
        self.system_prompt: Sequence[SystemContentBlockTypeDef] = (
            [system_prompt] if system_prompt else []
        )
        self.memory = memory
        self.inference_config = inference_config
        self.stream_messages = partial(
            stream_messages,
            client=self.client,
            model_id=self.model_id,
            system_prompt=self.system_prompt,
            memory=self.memory,
            inference_config=self.inference_config,
            stdout=False,
        )

    def _is_valid_message_order(self, new_messages: List[MessageUnionTypeDef]) -> bool:
        """Check if the new messages have a valid order.

        Messages must start with a user message and alternate
        between user and assistant.
        """
        if len(new_messages) == 1 and new_messages[0].get('role') == 'user':
            return True

        if new_messages[0].get('role') != 'user':
            return False

        for i, message in enumerate(new_messages[1:], start=1):
            if i % 2 == 0 and message.get('role') != 'user':
                return False
            if i % 2 == 1 and message.get('role') != 'assistant':
                return False

        return True

    @overload
    def send_messages(
        self, messages: List[MessageUnionTypeDef], streaming: Literal[True]
    ) -> Generator[
        tuple[ConverserStreamOutputTypeDefEnd, None | MessageUnionTypeDef], Any, Any
    ]: ...

    @overload
    def send_messages(
        self, messages: List[MessageUnionTypeDef], streaming: bool = False
    ) -> ConverseResponseTypeDef: ...

    @validate_message_order
    def send_messages(
        self, messages: List[MessageUnionTypeDef], streaming: bool = False
    ) -> Union[
        ConverseResponseTypeDef,
        Generator[tuple[ConverserStreamOutputTypeDefEnd, None | MessageUnionTypeDef], Any, Any],
    ]:
        """Send a message to the model.

        Args:
            messages (List[MessageUnionTypeDef]) : The messages to send.
            streaming (bool, optional): Whether to use streaming or not. Defaults to False.

        Returns:
            ConverseResponseTypeDef: The response from the model.

        Raises:
            ValueError: If the message order is invalid.
        """
        if self.memory:
            message_history = self.memory.get_history()
            messages = message_history + messages
        else:
            messages = messages

        if streaming:
            return cast(
                Generator[
                    tuple[ConverserStreamOutputTypeDefEnd, None | MessageUnionTypeDef], Any, Any
                ],
                self.stream_messages(messages=messages),
            )

        response: ConverseResponseTypeDef = self.client.converse(
            modelId=self.model_id,
            messages=messages,
            system=self.system_prompt,
            inferenceConfig=cast(
                InferenceConfigurationTypeDef, self.inference_config.model_dump()
            ),
        )

        match response['stopReason']:
            case 'end_turn' | 'tool_use' | 'max_tokens' | 'stop_sequence':
                content: List[ContentBlockTypeDef] = response['output']['message']['content']  # type: ignore - the keys are checked in the match statement
                if self.memory:
                    assistant_message: MessageUnionTypeDef = {
                        'role': 'assistant',
                        'content': content,
                    }
                    self.memory.add_messages([messages[-1]] + [assistant_message])
            # default case
            case _:
                raise NotImplementedError(
                    f"Stop reason '{response['stopReason']}' not implemented"
                )

        return response

    @overload
    def from_file(
        self,
        file_path: str,
        content_type: Literal['image', 'document'],
        streaming: Literal[True],
        user_text: str = 'Please describe the contents of the file in detail',
    ) -> Generator[
        tuple[ConverserStreamOutputTypeDefEnd, None | MessageUnionTypeDef], Any, Any
    ]: ...

    @overload
    def from_file(
        self,
        file_path: str,
        content_type: Literal['image', 'document'],
        streaming: bool = False,
        user_text: str = 'Please describe the contents of the file in detail',
    ) -> ConverseResponseTypeDef: ...

    def from_file(
        self,
        file_path: str,
        content_type: Literal['image', 'document'],
        streaming: bool = False,
        user_text: str = 'Please describe the contents of the file in detail',
    ) -> Union[
        ConverseResponseTypeDef,
        Generator[tuple[ConverserStreamOutputTypeDefEnd, None | MessageUnionTypeDef], Any, Any],
    ]:
        """Create a message from a file.

        Args:
            file_path (str): The path to the file.
            content_type (Literal['image', 'document']): The type of content in the file.
            user_text (Optional[str], optional): The user text to include with the file. Defaults to None.
            streaming (bool, optional): Whether to use streaming or not. Defaults to False.

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
        return self.send_messages([user_message], streaming=streaming)
