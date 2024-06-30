"""Memory class."""

from mypy_boto3_bedrock_runtime.type_defs import MessageOutputTypeDef, MessageTypeDef
from typing import List, Union


class Memory:
    """A class to store the message history."""

    def __init__(self) -> None:
        """Initialize the Memory class."""
        self.history: List[Union[MessageTypeDef, MessageOutputTypeDef]] = []

    def add_messages(self, messages: List[Union[MessageTypeDef, MessageOutputTypeDef]]) -> None:
        """Add a message to the history."""
        if not self._is_valid_message_order(messages):
            raise ValueError(
                'Invalid message order. Messages must start with a user message and alternate'
                'between user and assistant.'
            )
        self.history.extend(messages)

    def get_history(self) -> List[Union[MessageTypeDef, MessageOutputTypeDef]]:
        """Get the message history."""
        return self.history

    def _is_valid_message_order(
        self, new_message: Union[MessageTypeDef, MessageOutputTypeDef]
    ) -> bool:
        if not self.history:
            return new_message['role'] == 'user'
        last_message = self.history[-1]
        return last_message['role'] != new_message['role']

    def get_last_message(self) -> Union[MessageTypeDef, MessageOutputTypeDef]:
        """Get the last message in the history."""
        return self.history[-1]

    def clear_history(self) -> None:
        """Clear the message history."""
        self.history = []
