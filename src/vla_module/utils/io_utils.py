from pathlib import Path
from typing import TypeVar, Any
import json

T = TypeVar("T")

meta_directory="meta"
INFO_PATH = f"{meta_directory}/info.json"
STATS_PATH = f"{meta_directory}/stats.json"

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


def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_info(info: dict, local_dir: Path) -> None:
    write_json(info, local_dir / INFO_PATH)

def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary by joining keys with a separator.

    Example:
        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        >>> print(flatten_dict(dct))
        {'a/b': 1, 'a/c/d': 2, 'e': 3}

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key to prepend to the keys in this level.
        sep (str): The separator to use between keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    """Unflatten a dictionary with delimited keys into a nested dictionary.

    Example:
        flat_dct = {"a/b": 1, "a/c/d": 2, "e": 3}
        print(unflatten_dict(flat_dct))
        {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}

    Args:
        d (dict): A dictionary with flattened keys.
        sep (str): The separator used in the keys.

    Returns:
        dict: A nested dictionary.
    """
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict
