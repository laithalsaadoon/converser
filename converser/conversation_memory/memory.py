"""Memory class."""

from mypy_boto3_bedrock_runtime.type_defs import MessageUnionTypeDef
from typing import List


class Memory:
    """A class to store the message history."""

    def __init__(self) -> None:
        """Initialize the Memory class."""
        self.history: List[MessageUnionTypeDef] = []

    def add_messages(self, messages: List[MessageUnionTypeDef]) -> None:
        """Add a message to the history."""
        if not self._is_valid_message_history_order(messages):
            raise ValueError(
                'Invalid message order. Messages must start with a user message and alternate'
                ' between user and assistant.'
            )
        self.history.extend(messages)

    def get_history(self) -> List[MessageUnionTypeDef]:
        """Get the message history."""
        return self.history

    def _is_valid_message_history_order(self, new_messages: List[MessageUnionTypeDef]) -> bool:
        """Check if the new messages have a valid order.

        Messages must start with a user message and alternate
        between user and assistant.
        """
        message_history = self.get_history() + new_messages
        if message_history[0].get('role') != 'user':
            return False

        for i, message in enumerate(message_history[1:], start=1):
            if i % 2 == 0 and message.get('role') != 'user':
                return False
            if i % 2 == 1 and message.get('role') != 'assistant':
                return False

        return True

    def get_last_message(self) -> MessageUnionTypeDef:
        """Get the last message in the history."""
        return self.history[-1]

    def clear_history(self) -> None:
        """Clear the message history."""
        self.history = []
