# ruff: noqa: D100, D101, D103

import inspect
from mypy_boto3_bedrock_runtime.type_defs import ToolTypeDef
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from typing import Callable, get_type_hints


def generate_pydantic_model(func: Callable) -> type[BaseModel]:
    annotations = get_type_hints(func)
    fields = {}

    for param_name, param_type in annotations.items():
        if hasattr(param_type, '__metadata__') and param_type.__metadata__:
            annotated_type = param_type.__origin__
            field_info = param_type.__metadata__[0]
            fields[param_name] = (annotated_type, field_info)
        else:
            fields[param_name] = (param_type, Field())

    model = create_model(f'{func.__name__.capitalize()}InputModel', **fields)

    return model


def generate_tool_schema(func: Callable) -> ToolTypeDef:
    """Generate a ToolTypeDef schema from a function that uses Pydantic Field annotations.

    Args:
        func (Callable): The function for which to generate the model.

    Returns:
        ToolTypeDef: A ToolTypeDef schema for the function to use with Bedrock.

    Example:
        ```python
        from pydantic import Field, validate_call
        from typing import Annotated


        @validate_call
        def my_function(
            arg1: Annotated[int, Field(gt=10, description='The first number')],
            arg2: Annotated[str, Field(description='The second string')],
        ):
            '''This function takes two arguments, an integer and a string, and prints them.'''
            print(f'arg1: {arg1}, arg2: {arg2}')


        tool_schema = generate_tool_schema(my_function)
        ```
    """
    model = generate_pydantic_model(func)
    schema = model.model_json_schema()

    func_name = func.__name__
    func_doc = inspect.getdoc(func) or ''

    tool_schema: ToolTypeDef = {
        'toolSpec': {
            'name': func_name,
            'description': func_doc,
            'inputSchema': {'json': schema},
        }
    }

    return tool_schema


def parse_docstring(docstring: str):
    """Parse Google-style or NumPy-style docstrings.

    Separates the main description and extract parameter descriptions.
    """
    param_desc = {}
    param_section = False
    main_description = []
    lines = docstring.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('Args:') or line.startswith('Parameters'):
            param_section = True
            continue
        if param_section:
            if line and ':' in line:
                param_name, param_description = line.split(':', 1)
                param_name = param_name.split()[0]  # Get just the parameter name
                param_desc[param_name.strip()] = param_description.strip()
            elif not line:
                param_section = False
        else:
            main_description.append(line)

    main_description = '\n'.join(main_description).strip()
    return main_description, param_desc


def python_type_to_json_type(python_type):
    type_mapping = {
        'str': 'string',
        'int': 'integer',
        'float': 'number',
        'bool': 'boolean',
        'list': 'array',
        'dict': 'object',
    }
    return type_mapping.get(python_type, 'string')


def generate_json_schema(func: Callable) -> ToolTypeDef:
    func_name = func.__name__
    func_doc: str
    if inspect.getdoc(func):
        func_doc = str(inspect.getdoc(func))
        main_description, param_descriptions = parse_docstring(func_doc)
    else:
        raise ValueError(f'Function {func_name} does not have a docstring.')

    sig = inspect.signature(func)
    params = sig.parameters
    type_hints = get_type_hints(func)

    properties = {}
    required = []
    for param_name, param in params.items():
        param_type = type_hints.get(param_name, str)
        json_type = python_type_to_json_type(param_type.__name__)
        if param.default is param.empty:
            required.append(param_name)
        properties[param_name] = {
            'type': json_type,
            'description': param_descriptions.get(param_name, ''),
        }

    schema: ToolTypeDef = {
        'toolSpec': {
            'name': func_name,
            'description': main_description,
            'inputSchema': {
                'json': {'type': 'object', 'properties': properties, 'required': required}
            },
        }
    }

    return schema


def has_field_annotated_args(func):
    type_hints = get_type_hints(func, include_extras=True)
    return any(
        'Annotated' in str(arg_type) and isinstance(arg_type.__metadata__[0], FieldInfo)
        for arg_type in type_hints.values()
    )


# Generate a ToolTypeDef from a function.
# First, check if the function has Pydantic Fields .
# If it does, generate the schema using generate_tool_schema.
# If not, generate the schema using generate_json_schema.
# don't use try except... check if the function has Pydantic Field
def generate_tool_schema_from_function(func: Callable):
    if has_field_annotated_args(func):
        return generate_tool_schema(func)
    else:
        return generate_json_schema(func)
