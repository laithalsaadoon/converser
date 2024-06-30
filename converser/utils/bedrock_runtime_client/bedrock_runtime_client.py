"""This module provides a function to get a Bedrock client."""

from boto3 import _get_default_session
from boto3.session import Session
from botocore.config import Config
from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient


def get_bedrock_client(region: str, profile: str | None = None) -> BedrockRuntimeClient:
    """Get a Bedrock client.

    Args:
        region (str): The AWS region to use.
        profile (str, optional): The AWS profile to use. Defaults to None.

    Returns:
        BedrockRuntimeClient: The Bedrock client.

    """
    session: Session
    if profile is None:
        session = _get_default_session()
    else:
        session = Session(profile_name=profile)
    return session.client(
        'bedrock-runtime',
        region_name=region,
        config=Config(
            retries={
                'total_max_attempts': 20,
                'mode': 'adaptive',
            },
            read_timeout=300,
        ),
    )
