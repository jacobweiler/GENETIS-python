import sys
import logging
from pathlib import Path
import numpy as np
import shutil
from datetime import datetime
from ruamel.yaml import YAML
import random
from typing import Union

log = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Union[str, Path] = None):
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    logging.debug(f"Logging initialized at {log_level} level")
    if log_file:
        logging.debug(f"Logging to file: {log_file}")


def init(run_name: Union[str, Path], settings_path: Union[str, Path] = "settings.yaml"):
    workingdir = Path.cwd()
    run_dir = workingdir / "Run_Outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "job_outs" / "xf_out").mkdir(parents=True, exist_ok=True, mode=0o775)
    (run_dir / "job_outs" / "xf_err").mkdir(parents=True, exist_ok=True, mode=0o775)
    (run_dir / "job_outs" / "ara_out").mkdir(parents=True, exist_ok=True, mode=0o775)
    (run_dir / "job_outs" / "ara_err").mkdir(parents=True, exist_ok=True, mode=0o775)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True, mode=0o775)
    (run_dir / "xmacros").mkdir(parents=True, exist_ok=True, mode=0o775)

    # Copy settings.yaml to run_dir
    settings_path = Path(settings_path)
    if not settings_path.exists():
        log.error(f"settings file {settings_path} not found. Skipping copy.")
        sys.exit(1)

    dest_settings = run_dir / "settings.yaml"
    shutil.copy(settings_path, dest_settings)
    log.info(f"Copied settings file to {dest_settings}")

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(dest_settings, "r") as f:
        settings_data = yaml.load(f)

    xf_proj = run_dir / f"{run_name}.xf"
    xmacros_dir = run_dir / "xmacros"
    global_xmacros = workingdir / "src/xf"
    ara_scripts = workingdir / "src/ara"
    settings_data["workingdir"] = str(workingdir.resolve())
    settings_data["run_name"] = run_name
    settings_data["run_dir"] = str(run_dir.resolve())
    settings_data["xf_proj"] = str(xf_proj.resolve())
    settings_data["run_xmacros"] = str(xmacros_dir.resolve())
    settings_data["xmacros"] = str(global_xmacros.resolve())
    settings_data["ara_scripts"] = str(ara_scripts.resolve())

    freq_start = settings_data.get("freq_start", 83.33)
    freq_step = settings_data.get("freq_step", 16.67)
    freq_num = settings_data.get("freq_num", 60)
    freq_vals = [
        83.33, 100.00, 116.67, 133.33, 150.00, 166.67, 183.34, 200.00, 216.67, 233.34,
        250.00, 266.67, 283.34, 300.00, 316.67, 333.34, 350.00, 366.67, 383.34, 400.01,
        416.67, 433.34, 450.01, 466.67, 483.34, 500.01, 516.68, 533.34, 550.01, 566.68,
        583.34, 600.01, 616.68, 633.34, 650.01, 666.68, 683.35, 700.01, 716.68, 733.35,
        750.01, 766.68, 783.35, 800.01, 816.68, 833.35, 850.02, 866.68, 883.35, 900.02,
        916.68, 933.35, 950.02, 966.68, 983.35, 1000.00, 1016.70, 1033.40, 1050.00, 1066.70
    ]
    settings_data["freq_vals"] = freq_vals

    # Add or replace rng_seed only if missing or set to "random"
    if "rng_seed" not in settings_data or settings_data["rng_seed"] == "random":
        new_seed = random.randint(0, 2**32 - 1)
        settings_data["rng_seed"] = new_seed
        log.info(f"Generated new rng_seed: {new_seed}")

    with open(dest_settings, "w") as f:
        yaml.dump(settings_data, f)

    log.info("Updated settings.yaml with run_dir and rng_seed")

    # Save current date to start_date.txt
    start_date_path = run_dir / "start_date.txt"
    with open(start_date_path, "w") as f:
        f.write(datetime.now().isoformat())
    log.info(f"Start date written to {start_date_path}")

    log.info(f"Run directory initialized at {run_dir.resolve()}")
