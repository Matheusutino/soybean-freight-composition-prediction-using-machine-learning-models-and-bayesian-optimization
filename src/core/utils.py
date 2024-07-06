import json
import yaml
from pathlib import Path
from typing import List, Optional

def write_json(file_path: str, data: List[Optional[str]]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def write_yaml(file_path: str, data: List[Optional[str]]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)

def read_json(file_path: str) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_yaml(file_path: str) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_lines_csv(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    return lines

def create_folder(folder_path: str) -> None:
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)