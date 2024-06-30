"""Rename a file to a sanitized version of its name."""

import re
from pathlib import Path


def sanitize_file_name(file_path) -> str:
    """Rename a file to a sanitized version of its name.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The sanitized file name.
    """
    # Extract the file name from the file path
    file_name = Path(file_path).name

    # Replace any character that is not alphanumeric, whitespace, dash,
    # hyphen, parentheses, or square brackets with an underscore
    sanitized_name = re.sub(r'[^a-zA-Z0-9\s\-\(\)\[\]]', '_', file_name)

    # Replace multiple consecutive whitespace characters with a single space
    sanitized_name = re.sub(r'\s+', ' ', sanitized_name)

    return sanitized_name
