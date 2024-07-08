"""Projen configuration file for the converser project."""

from projen import ProjectType
from projen.python import PythonProject


project = PythonProject(
    author_email='alsaadoonlaith@gmail.com',
    author_name='Laith Al-Saadoon',
    module_name='converser',
    name='converser',
    version='0.1.0',
    project_type=ProjectType.LIB,
    pip=False,
    venv=False,
    setuptools=False,
    poetry=True,
    deps=[
        'boto3',
        'botocore',
        'pydantic',
        'requests',
        'tqdm',
        "boto3-stubs@{version = '*', extras = ['bedrock-runtime', 'bedrock']}",
    ],
    dev_deps=[
        'beautifulsoup4',
        'jinja2',
        'ruff',
        'bandit',
        'mypy',
        'pytest',
        'pytest-cov',
    ],
)

if (pyproject_toml := project.try_find_object_file('pyproject.toml')) is None:
    raise ValueError('pyproject.toml not found')

tool_ruff = pyproject_toml.add_override(
    'tool.ruff',
    {
        'line-length': 99,
    },
)

tool_ruff_lint = pyproject_toml.add_override(
    'tool.ruff.lint',
    {
        'select': [
            'C',
            'D',
            'E',
            'F',
            'I',
            'W',
        ],
    },
)

tool_ruff_lint_isort = pyproject_toml.add_override(
    'tool.ruff.lint.isort',
    {
        'lines-after-imports': 2,
        'no-sections': True,
    },
)

tool_ruff_lint_pydocstyle = pyproject_toml.add_override(
    'tool.ruff.lint.pydocstyle',
    {
        'convention': 'google',
    },
)

tool_ruff_format = pyproject_toml.add_override(
    'tool.ruff.format',
    {
        'quote-style': 'single',
        'indent-style': 'space',
        'skip-magic-trailing-comma': False,
        'line-ending': 'auto',
        'docstring-code-format': True,
    },
)

project.add_task('lint', exec='poetry run ruff check . --verbose $@', receive_args=True)
project.add_task('format', exec='poetry run ruff format .')

if (test_task := project.tasks.try_find('test')) is not None:
    test_task.reset('poetry run pytest $@', receive_args=True)

project.synth()
