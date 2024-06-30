"""Test the Converse class."""

import pytest
from converser import Converse, InferenceConfig, ModelId


# Define the test cases
@pytest.mark.parametrize(
    'model_id, system_prompt, memory, inference_config, region',
    [
        (ModelId.CLAUDE_3_SONNET, None, None, InferenceConfig(), 'us-west-2'),
    ],
)
def test_converse(
    model_id: ModelId,
    inference_config: InferenceConfig,
    region: str,
):
    """Test the Converse class."""
    # Create a new Converse object
    converse = Converse(
        model_id=model_id,
        inference_config=inference_config,
        region=region,
    )

    # Test the Converse object
    assert converse.model_id == model_id
    assert converse.inference_config == inference_config
    assert converse.client is not None
    assert converse.stream_messages is not None
    assert converse.send_message is not None
