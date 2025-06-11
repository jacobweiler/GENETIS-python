# Functions related to initializing loop and/or a new run
import sys
import logging
import yaml
from pathlib import Path
import shutil
from datetime import datetime

log = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str | Path = None):
    """
    Sets up logging configuration.

    Args:
        log_level (str): Logging verbosity level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file (str | Path, optional): If provided, logs will be written to this file as well.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Clear any existing handlers (important if running multiple times)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.StreamHandler()]  # Console output

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        handlers.append(logging.FileHandler(log_path, mode="a"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers
    )

    logging.debug(f"Logging initialized at {log_level} level")
    if log_file:
        logging.debug(f"Logging to file: {log_file}")


def init(rundir_path: str | Path, config_path: str | Path = "config.yml"):
    rundir = Path(rundir_path)
    rundir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (rundir / "job_outs" / "xf_out").mkdir(parents=True, exist_ok=True)
    (rundir / "job_outs" / "xf_err").mkdir(parents=True, exist_ok=True)
    (rundir / "job_outs" / "ara_out").mkdir(parents=True, exist_ok=True)
    (rundir / "job_outs" / "ara_err").mkdir(parents=True, exist_ok=True)
    (rundir / "plots").mkdir(parents=True, exist_ok=True)
    (rundir / "generation_data").mkdir(parents=True, exist_ok=True)

    # Copy config.yml to rundir
    config_path = Path(config_path)
    if config_path.exists():
        dest_config = rundir / "config.yml"
        shutil.copy(config_path, dest_config)
        log.info(f"Copied config file to {dest_config}")

        with open(dest_config, "r") as f:
            config_data = yaml.safe_load(f)

        if "run_params" not in config_data or config_data["run_params"] is None:
            config_data["run_params"] = {}

        config_data["run_params"]["rundir"] = str(rundir.resolve())

        with open(dest_config, "w") as f:
            yaml.safe_dump(config_data, f)

        log.info(f"Updated config.yml with rundir path under run_params")
    else:
        log.error(f"Config file {config_path} not found. Skipping copy.")
        sys.exit(1)

    # Save current date to start_date.txt
    start_date_path = rundir / "start_date.txt"
    with open(start_date_path, "w") as f:
        f.write(datetime.now().isoformat())
    log.info(f"Start date written to {start_date_path}")

    log.info(f"Run directory initialized at {rundir.resolve()}")