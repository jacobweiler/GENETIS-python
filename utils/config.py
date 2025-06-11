# config related functions
import yaml

CONFIG = {}


def load_config(path="config.yml"):
    """
    Loads a YAML config from the given path into the global CONFIG variable.

    Args:
        path (str): Path to the config.yml file.
    """
    global CONFIG
    with open(path, "r") as f:
        CONFIG = yaml.safe_load(f)


def run():
    return CONFIG.get("run", {})


def sim():
    return CONFIG.get("sim", {})


def ga():
    return CONFIG.get("ga", {})


def antenna():
    return CONFIG.get("antennas", {})


def vpol():
    return CONFIG.get("antennas", {}).get("vpol", {})


def hpol():
    return CONFIG.get("antennas", {}).get("hpol", {})


def job():
    return CONFIG.get("job", {})
