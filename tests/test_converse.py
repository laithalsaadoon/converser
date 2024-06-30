"""Test the Converse class."""

import pytest
from converser import Converse, InferenceConfig, Memory, ModelId
from mypy_boto3_bedrock_runtime.type_defs import SystemContentBlockTypeDef
from typing import Optional


# Define the test cases
@pytest.mark.parametrize(
    'model_id, system_prompt, memory, inference_config, region',
    [
        (ModelId.CLAUDE_3_SONNET, None, None, InferenceConfig(), 'us-west-2'),
    ],
)
def test_converse(
    model_id: ModelId,
    system_prompt: Optional[SystemContentBlockTypeDef],
    memory: Optional[Memory],
    inference_config: InferenceConfig,
    region: str,
):
    """Test the Converse class."""
    # Create a new Converse object
    converse = Converse(
        model_id=model_id,
        system_prompt=system_prompt,
        memory=memory,
        inference_config=inference_config,
        region=region,
    )

    # Test the Converse object
    assert converse.model_id == model_id
    assert converse.system_prompt == ([system_prompt] if system_prompt else [])
    assert converse.memory == memory
    assert converse.inference_config == inference_config
    assert converse.client is not None
    assert converse.stream_messages is not None
    assert converse.send_messages is not None


def test_send_message():
    """Test the send_message method of the Converse class."""
    # Create a new Converse object
    converse = Converse(
        model_id=ModelId.CLAUDE_3_SONNET,
        inference_config=InferenceConfig(),
        region='us-west-2',
    )

    # Define the messages to send
    messages = [
        {'role': 'user', 'content': [{'text': 'Hello'}]},
    ]

    # Call the send_message method
    response = converse.send_messages(messages)

    # Assert the response
    assert 'stopReason' in response
    assert 'output' in response
    assert 'message' in response['output']
    assert 'content' in response['output']['message']
    assert response['output']['message']['role'] == 'assistant'


def test_from_file():
    """Test the from_file method of the Converse class."""
    # Create a new Converse object
    converse = Converse(
        model_id=ModelId.CLAUDE_3_SONNET,
        inference_config=InferenceConfig(),
        region='us-west-2',
    )

    # Define the file path and content type
    file_path = './tests/test_file.txt'
    content_type = 'document'

    # Call the from_file method
    response = converse.from_file(file_path, content_type)

    # Assert the response
    assert 'stopReason' in response
    assert 'output' in response
    assert 'message' in response['output']
    assert 'content' in response['output']['message']
