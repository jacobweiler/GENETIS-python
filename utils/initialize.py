# Functions related to initializing loop and/or a new run
import sys
import logging
from pathlib import Path
import shutil
from datetime import datetime

log = logging.getLogger(__name__)


def setup_logging(log_level):
    """
    Sets config for logging

    Args:
        log_level (str): level of verbosity
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def init(rundir_path: str | Path, config_path: str | Path = "config.yml"):
    """
    Initializes the loop to create all needed files + directories for a new run

    Args:
        rundir_path (Path): Location of run directory
        config_path (Path): Location of the config file
    """
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
        shutil.copy(config_path, rundir / "config.yml")
        log.info(f"Copied config file to {rundir / 'config.yml'}")
    else:
        log.error(f"Config file {config_path} not found. Skipping copy.")
        sys.exit(1)

    # Save current date to start_date.txt
    start_date_path = rundir / "start_date.txt"
    with open(start_date_path, "w") as f:
        f.write(datetime.now().isoformat())
    log.info(f"Start date written to {start_date_path}")

    log.info(f"Run directory initialized at {rundir.resolve()}")
