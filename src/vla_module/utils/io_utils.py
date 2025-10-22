from pathlib import Path
from typing import TypeVar, Any
import json

T = TypeVar("T")


def load_json(fpath: Path) -> dict:
    """Load JSON data from file."""
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


def validate_and_convert_type(target: Any, source: Any, path: str = "root") -> Any:
    """
    Validate source matches target's type and convert if needed.
    
    Args:
        target: Template object defining expected structure and types
        source: Data to validate and convert
        path: Current path in the structure (for error messages)
        
    Returns:
        Converted value matching target's type
    """
    target_type = type(target)
    source_type = type(source)
    
    # Handle dict
    if isinstance(target, dict):
        if not isinstance(source, dict):
            raise TypeError(f"At {path}: expected dict, got {source_type.__name__}")
        
        if target.keys() != source.keys():
            raise ValueError(
                f"At {path}: key mismatch.\n"
                f"Expected: {set(target.keys())}\n"
                f"Got: {set(source.keys())}"
            )
        
        return {
            key: validate_and_convert_type(target[key], source[key], f"{path}.{key}")
            for key in target
        }
    
    # Handle list
    elif isinstance(target, list):
        if not isinstance(source, list):
            raise TypeError(f"At {path}: expected list, got {source_type.__name__}")
        
        if len(target) != len(source):
            raise ValueError(
                f"At {path}: length mismatch (expected {len(target)}, got {len(source)})"
            )
        
        return [
            validate_and_convert_type(target[i], source[i], f"{path}[{i}]")
            for i in range(len(target))
        ]
    
    # Handle tuple (JSON stores as list)
    elif isinstance(target, tuple):
        if not isinstance(source, list):
            raise TypeError(f"At {path}: expected list (for tuple), got {source_type.__name__}")
        
        if len(target) != len(source):
            raise ValueError(
                f"At {path}: length mismatch (expected {len(target)}, got {len(source)})"
            )
        
        return tuple(
            validate_and_convert_type(target[i], source[i], f"{path}[{i}]")
            for i in range(len(target))
        )
    
    # Handle primitives (int, float, str, bool, None)
    else:
        if target_type is not source_type:
            raise TypeError(
                f"At {path}: expected {target_type.__name__}, got {source_type.__name__}"
            )
        return source


def deserialize_json_into_object(fpath: Path, obj: T) -> T:
    """
    Load JSON from file and validate it matches the structure and types of obj.
    
    Args:
        fpath: Path to JSON file
        obj: Template object defining expected structure
        
    Returns:
        New object with same structure as obj, filled with data from JSON
        
    Example:
        >>> template = {"config": {"lr": 0.001, "epochs": 100}, "mode": "train"}
        >>> loaded = deserialize_json_into_object(Path("config.json"), template)
    """
    data = load_json(fpath)
    return validate_and_convert_type(obj, data)
