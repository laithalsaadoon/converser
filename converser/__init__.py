"""The main package for the project."""

from .conversation_memory.memory import Memory
from .converse import Converse
from .models.models import InferenceConfig, ModelId
from converser.utils import get_bedrock_client


# Define the public API of the package
__all__ = [
    'Converse',
    'Memory',
    'get_bedrock_client',
    'InferenceConfig',
    'ModelId',
]
