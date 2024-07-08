"""Model capabilities checker."""

import boto3
import concurrent.futures
from functools import partial
from jinja2 import Template
from jinja2.loaders import FileSystemLoader
from jinja2.sandbox import SandboxedEnvironment
from mypy_boto3_bedrock import BedrockClient
from mypy_boto3_bedrock.type_defs import FoundationModelSummaryTypeDef
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime.type_defs import (
    DocumentBlockTypeDef,
    GuardrailConfigurationTypeDef,
    ImageBlockTypeDef,
    SystemContentBlockTypeDef,
    ToolConfigurationTypeDef,
)
from typing import Callable, Sequence, TypedDict, TypeGuard


runtime_client: BedrockRuntimeClient = boto3.client('bedrock-runtime')
environment = SandboxedEnvironment(
    loader=FileSystemLoader(
        'codegen',
    )
)
template: Template = environment.get_template('model_ids_template.j2')


class ModelInformation(TypedDict):
    """A dictionary containing information about a model."""

    modelLifecycle: dict
    modelName: str
    modelId: str


class ModelFunctionality(TypedDict):
    """A dictionary containing the results of the functionality checks."""

    model_name: str
    model_id: str
    converse: bool
    converse_stream: bool
    system_prompts: bool
    document_chat: bool
    vision: bool
    tool_use: bool
    streaming_tool_use: bool
    guardrails: bool


def try_except(func: Callable, **kwargs) -> bool:
    """Try to execute a function and return True if it succeeds, False otherwise.

    Args:
        func (Callable): The function to execute. Should be one of bedrock-runtime's converse or
        converser_stream
        **kwargs: The keyword arguments to pass through to converse or converse_stream.

    Returns:
        bool: True if the function executes successfully, False otherwise.
    """
    try:
        func(**kwargs)
        return True
    except Exception:
        return False


def get_active_model_summaries() -> list[ModelInformation]:
    """Get a list of active model summaries."""

    def _is_active_model(model: FoundationModelSummaryTypeDef) -> TypeGuard[ModelInformation]:
        return (
            'modelLifecycle' in model
            and 'modelName' in model
            and isinstance(model['modelLifecycle'], dict)
            and model['modelLifecycle'].get('status') == 'ACTIVE'
        )

    client_pdw: BedrockClient = boto3.client('bedrock', region_name='us-west-2')
    client_iad: BedrockClient = boto3.client('bedrock', region_name='us-east-1')

    models_pre_filter_pwd = client_pdw.list_foundation_models(
        byOutputModality='TEXT',
        byInferenceType='ON_DEMAND',
    )['modelSummaries']

    models_pre_filter_iad = client_iad.list_foundation_models(
        byOutputModality='TEXT',
        byInferenceType='ON_DEMAND',
    )['modelSummaries']

    # combine and remove duplicates
    models_pre_filter = list(
        {
            model['modelId']: model for model in models_pre_filter_pwd + models_pre_filter_iad
        }.values()
    )

    return [model for model in models_pre_filter if _is_active_model(model)]


def check_functionality(model_config: ModelInformation) -> ModelFunctionality:
    """Check the functionality of a model by performing various conversational tasks.

    Args:
        model_config (ModelInformation): Information about the model.

    Returns:
        Functionality: A dictionary containing the results of the functionality checks.

    """
    tool_config: ToolConfigurationTypeDef = {
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
                                    you want the most popular song. Example calls
                                    signs are WZPZ and WKRP.""",
                                }
                            },
                            'required': ['sign'],
                        }
                    },
                }
            }
        ]
    }
    model_id = model_config['modelId']
    model_name = model_config['modelName']

    # try to use the model with different capabilities of the converse API
    converse_partial = partial(
        runtime_client.converse,
        modelId=model_id,
        messages=[{'role': 'user', 'content': [{'text': 'What is the most popular song?'}]}],
        inferenceConfig={
            'temperature': 0.01,
            'maxTokens': 10,
        },
    )
    converse_stream_partial = partial(
        runtime_client.converse_stream,
        modelId=model_id,
        messages=[{'role': 'user', 'content': [{'text': 'What is the most popular song?'}]}],
        inferenceConfig={
            'temperature': 0.01,
            'maxTokens': 10,
        },
    )

    system_prompts: Sequence[SystemContentBlockTypeDef] = [{'text': 'You are a helpful AI.'}]

    document: DocumentBlockTypeDef = {
        'format': 'pdf',
        'name': 'document',
        'source': {'bytes': open('codegen/sample.pdf', 'rb').read()},
    }

    image: ImageBlockTypeDef = {
        'format': 'png',
        'source': {'bytes': open('codegen/sample.png', 'rb').read()},
    }

    # override for sonnet and opus since they may not be available in the account or region
    if any(x in model_id.lower() for x in ['sonnet', 'opus']):
        return {
            'model_name': model_name,
            'model_id': model_id,
            'converse': True,
            'converse_stream': True,
            'system_prompts': True,
            'document_chat': True,
            'vision': True,
            'tool_use': True,
            'streaming_tool_use': True,
            'guardrails': True,
        }

    return {
        'model_name': model_name,
        'model_id': model_id,
        'converse': try_except(converse_partial),
        'converse_stream': try_except(converse_stream_partial),
        'system_prompts': try_except(converse_partial, system=system_prompts),
        'document_chat': try_except(
            runtime_client.converse,
            modelId=model_id,
            messages=[
                {
                    'role': 'user',
                    'content': [{'text': 'summarize the document.'}, {'document': document}],
                }
            ],
        ),
        'vision': try_except(
            runtime_client.converse,
            modelId=model_id,
            messages=[
                {
                    'role': 'user',
                    'content': [{'text': 'What is in this image?'}, {'image': image}],
                }
            ],
        ),
        'tool_use': try_except(converse_partial, toolConfig=tool_config),
        'streaming_tool_use': try_except(converse_stream_partial, toolConfig=tool_config),
        'guardrails': try_except(converse_partial, guardrailConfig=guardrail),
    }


guardrail: GuardrailConfigurationTypeDef = {
    'guardrailIdentifier': '5yc6ysthwkb0',
    'guardrailVersion': '1',
}


active_models = get_active_model_summaries()


def concurrent_process(model: ModelInformation):
    """Dump the model capabilities to a JSON file."""
    return check_functionality(model)


with concurrent.futures.ThreadPoolExecutor() as executor:
    result = executor.map(concurrent_process, active_models)

with open('converser/models/model_ids.py', 'w') as f:
    f.write(template.render(models=list(result)))
