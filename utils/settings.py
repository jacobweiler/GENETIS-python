# settings file functions
import yaml

SETTINGS = {}


def load_settings(path="settings.yaml"):
    """
    Loads a YAML settings from the given path into the global settings variable.
    """
    global SETTINGS
    with open(path, "r") as f:
        SETTINGS = yaml.safe_load(f)


def get(key, default=None):
    """
    Retrieve a value from the global settings.

    Args:
        key (str): The key to look up.
        default: Value to return if key is not found.

    Returns:
        The value associated with the key, or default.
    """
    return SETTINGS.get(key, default)


def all():
    """
    Return the entire settings dictionary.
    """
    return SETTINGS
