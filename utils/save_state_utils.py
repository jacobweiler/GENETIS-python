# Savestate-related functions
import os
import yaml


def initialize_state():
    return {
        "generation": 0,
        "step": "genes",  # Goes 'genes', 'xf', 'ara', 'plot' in loop
    }


def get_current_state(filename):
    """
    Load or initialize a savestate from the given file.

    Args:
        filename (str): Path to the savestate YAML file.

    Returns:
        dict: The current state dictionary.
    """
    if os.path.exists(filename):
        print("Loading existing savestate...")
        state = load_state(filename)
    else:
        print("No savestate found, initializing new state...")
        state = initialize_state()
        save_state(state, filename)
    return state


def save_state(state_obj, filename):
    with open(filename, "w") as f:
        yaml.dump(state_obj, f, sort_keys=False)


def load_state(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)
