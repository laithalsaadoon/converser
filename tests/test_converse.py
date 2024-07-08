"""Test the Converse class."""

import pytest
from converser import Converse, InferenceConfig
from converser.conversation_memory.memory import Memory
from converser.models import model_ids
from converser.tool_use.tool_use import generate_tool_schema_from_function
from mypy_boto3_bedrock_runtime.type_defs import (
    ConverseResponseTypeDef,
    InferenceConfigurationTypeDef,
    MessageUnionTypeDef,
    ToolConfigurationTypeDef,
    ToolTypeDef,
)
from typing import List, cast


# upstream boto3 has this warning
pytestmark = pytest.mark.filterwarnings('ignore:datetime.datetime.utcnow()')


# Define the test cases
@pytest.fixture
def converse_args():
    """Standard arguments for the Converse class."""
    return {
        'model_id': model_ids.ANTHROPIC_CLAUDE_3_HAIKU_20240307_V1_0['model_id'],
        'system_prompt': None,
        'memory': None,
        'inference_config': InferenceConfig(
            temperature=0.01,
            maxTokens=10,
        ),
        'region': 'us-east-1',
    }


@pytest.fixture
def converse_with_memory_args():
    """Arguments for the Converse class with memory."""
    return {
        'model_id': model_ids.ANTHROPIC_CLAUDE_3_HAIKU_20240307_V1_0['model_id'],
        'system_prompt': None,
        'memory': Memory(),
        'inference_config': InferenceConfig(
            temperature=0.01,
            maxTokens=10,
        ),
        'region': 'us-east-1',
    }


def test_converse(converse_args):
    """Test the Converse class."""
    # Create a new Converse object
    converse = Converse(**converse_args)

    # Test the Converse object
    assert converse.client is not None
    assert converse.stream_messages is not None
    assert converse.send_messages is not None


def test_send_message(converse_args):
    """Test the send_message method of the Converse class."""
    # Create a new Converse object
    converse = Converse(**converse_args)

    # Define the messages to send
    messages: List[MessageUnionTypeDef] = [
        {'role': 'user', 'content': [{'text': 'Hello'}]},
    ]

    # Call the send_message method
    response: ConverseResponseTypeDef = converse.send_messages(messages)

    # Assert the response
    assert 'stopReason' in response
    assert 'output' in response
    assert 'message' in response['output']
    assert 'content' in response['output']['message']
    assert response['output']['message']['role'] == 'assistant'


def test_send_messages(converse_args):
    """Test with multiple messages. "Putting" words in claude's mouth."""
    converse = Converse(**converse_args)

    messages: List[MessageUnionTypeDef] = [
        {'role': 'user', 'content': [{'text': 'Hello'}]},
        {'role': 'assistant', 'content': [{'text': 'Hi there! My name is'}]},
    ]

    response: ConverseResponseTypeDef = converse.send_messages(messages)

    assert 'stopReason' in response
    assert 'output' in response
    assert 'message' in response['output']
    assert 'content' in response['output']['message']
    assert response['output']['message']['role'] == 'assistant'


def test_send_conversation(converse_args):
    """Test sending a conversation."""
    # Create a new Converse object
    converse = Converse(**converse_args)

    # Define the conversation to send
    conversation: List[MessageUnionTypeDef] = [
        {'role': 'user', 'content': [{'text': 'Hello'}]},
        {'role': 'assistant', 'content': [{'text': 'Hi there!'}]},
        {'role': 'user', 'content': [{'text': 'How are you?'}]},
        {'role': 'assistant', 'content': [{'text': 'I am doing well, thank you.'}]},
        {'role': 'user', 'content': [{'text': 'That is good to hear.'}]},
        {'role': 'assistant', 'content': [{'text': 'Yes, it is.'}]},
        {'role': 'user', 'content': [{'text': 'Goodbye.'}]},
    ]

    response: ConverseResponseTypeDef = converse.send_messages(conversation)

    assert 'stopReason' in response
    assert 'output' in response
    assert 'message' in response['output']
    assert 'content' in response['output']['message']
    assert response['output']['message']['role'] == 'assistant'


def test_from_file(converse_args):
    """Test the from_file method of the Converse class."""
    # Create a new Converse object
    converse = Converse(**converse_args)

    # Define the file path and content type
    file_path = './tests/test_file.txt'
    content_type = 'document'

    # Call the from_file method
    response: ConverseResponseTypeDef = converse.from_file(file_path, content_type)

    # Assert the response
    assert 'stopReason' in response
    assert 'output' in response
    assert 'message' in response['output']
    assert 'content' in response['output']['message']


def test_messages_out_of_order(converse_args):
    """Test sending messages out of order."""
    # Create a new Converse object
    converse = Converse(**converse_args)

    # Define the messages to send
    messages: List[MessageUnionTypeDef] = [
        {'role': 'assistant', 'content': [{'text': 'Hello'}]},
        {'role': 'user', 'content': [{'text': 'Hello'}]},
        {'role': 'assistant', 'content': [{'text': 'Hello'}]},
        {'role': 'assistant', 'content': [{'text': 'Hello'}]},
    ]

    # Call the send_message method
    with pytest.raises(ValueError):
        # print('The message order is invalid and is intentionally raising an error.')
        converse.send_messages(messages)


def test_memory(converse_with_memory_args):
    """Test the memory attribute of the Converse class."""
    # Create a new Converse object
    converse = Converse(**converse_with_memory_args)

    # Define the messages to send
    messages: List[MessageUnionTypeDef] = [
        {'role': 'user', 'content': [{'text': 'Hello'}]},
    ]

    # Call the send_message method
    converse.send_messages(messages)

    # add another message
    converse.send_messages([{'role': 'user', 'content': [{'text': 'How are you?'}]}])

    # Assert the memory attribute
    assert converse.memory is not None
    assert len(converse.memory.get_history()) == 4
    assert converse.memory.get_last_message()['role'] == 'assistant'


def test_streaming(converse_args):
    """Test the streaming attribute of the Converse class."""
    # Create a new Converse object
    converse = Converse(**converse_args)

    # Define the messages to send
    messages: List[MessageUnionTypeDef] = [
        {'role': 'user', 'content': [{'text': 'Hello'}]},
    ]

    # Call the send_message method
    response = converse.send_messages(messages, True)

    # Assert the response
    assert response is not None
    for event, final_message in response:
        assert 'done' in event
        if event['done']:
            assert final_message is not None
            break


def test_tooluse(converse_args):
    """Test the tool use functionality."""
    tools: ToolConfigurationTypeDef = {
        'tools': [
            {
                'toolSpec': {
                    'name': 'top_song',
                    'description': 'Get the most popular song played on a radio station.',
                    'inputSchema': {
                        'json': {
                            'type': 'object',
                            'properties': {
                                'sign': {
                                    'type': 'string',
                                    'description': """The call sign for the radio station for which
                                    you want the most popular song.""",
                                }
                            },
                            'required': ['sign'],
                        }
                    },
                }
            }
        ]
    }

    converse = Converse(**converse_args)
    client = converse.client
    model_id = converse.model_id
    inference_config = cast(InferenceConfigurationTypeDef, converse.inference_config.model_dump())
    inference_config['maxTokens'] = 100
    messages: List[MessageUnionTypeDef] = [
        {'role': 'user', 'content': [{'text': 'Whats the most popular song on KIIS?'}]},
    ]

    response = client.converse(
        modelId=model_id, inferenceConfig=inference_config, messages=messages, toolConfig=tools
    )

    assert response
    assert response['stopReason']
    assert response['stopReason'] == 'tool_use'

    assert response['output']
    assert response['output']['message']['content'][0]['toolUse']  # type: ignore
    assert response['output']['message']['content'][0]['toolUse']['name'] == 'top_song'  # type: ignore
    assert response['output']['message']['content'][0]['toolUse']['input']['sign'] == 'KIIS'  # type: ignore


def test_tool_schema_generation_docstring():
    """Test the generation of tool schemas."""

    def top_song(sign: str):
        """Get the most popular song played on a radio station.

        Args:
            sign (str): The call sign for the radio station
            for which you want the most popular song.
            Example call signs are WZPZ, WKRP, KIIS.
        """
        print(f'The top song for {sign} is "Never Gonna Give You Up" by Rick Astley.')

    tool_schema = generate_tool_schema_from_function(top_song)

    assert tool_schema
    assert cast(ToolTypeDef, tool_schema)


def test_tool_schema_generation_pydantic():
    """Test the generation of tool schemas."""
    from pydantic import Field, validate_call
    from typing import Annotated

    @validate_call
    def func_with_pydantic_fields(
        arg1: Annotated[int, Field(gt=10, description='The first number')],
        arg2: Annotated[str, Field(description='The second string')],
    ):
        """This function takes two arguments, an integer and a string, and prints them.

        Args:
            arg1 (int): The first number, which must be greater than 10.
            arg2 (str): The second string.
        """
        print(f'arg1: {arg1}, arg2: {arg2}')

    tool_schema = generate_tool_schema_from_function(func_with_pydantic_fields)
    assert cast(ToolTypeDef, tool_schema)
    assert tool_schema
