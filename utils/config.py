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
    return CONFIG.get("run_params", {})


def sim():
    return CONFIG.get("sim_params", {})


def ga():
    return CONFIG.get("ga_params", {})


def antenna():
    return CONFIG.get("antenna_params", {})


def vpol():
    return CONFIG.get("antenna_params", {}).get("vpol_params", {})


def hpol():
    return CONFIG.get("antenna_params", {}).get("hpol_params", {})


def job():
    return CONFIG.get("job_submission", {})
