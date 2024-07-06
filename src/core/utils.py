import json
from pathlib import Path

def write_json(file_path: str, data: dict) -> None:
    """
    Write a dictionary to a JSON file.

    Args:
        file_path (str): The path to the file where the JSON data will be written.
        data (dict): The dictionary to be written as JSON.

    Raises:
        TypeError: If the provided data is not a dictionary.
        IOError: If there is an error writing the file.
    """
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary.")
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        raise IOError(f"Error writing file {file_path}: {e}")

def read_json(file_path: str) -> dict:
    """
    Read a JSON file and return its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file to be read.

    Returns:
        dict: The contents of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON from file {file_path}: {e}")

def create_folder(folder_path: str) -> None:
    """
    Create a folder and any necessary parent directories.

    Args:
        folder_path (str): The path to the folder to be created.

    Raises:
        OSError: If there is an error creating the directory.
    """
    try:
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Error creating directory {folder_path}: {e}")
