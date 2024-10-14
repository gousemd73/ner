"""
fileio.py
provides file input output operations utility
"""

import json

def read_file(path: str) -> str:
    """
    reads file from given path
    """

    with open(path,'r+') as f:
        file_content = f.read()

    return file_content

def write_file(file_content: str, path: str) -> None:

    """
    writes file_content to given path
    """

    with open(path,'w+') as f:
        f.write(file_content)

def read_json(path: str) -> str:

    """
    reads json file from given path
    """
    try:
        with open(path) as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return []
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file.")
        return []