# GENETIS Main Loop
import argparse
from pathlib import Path

from utils import config, initialize
import utils.save_state_utils as save_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Run Specific Details")
    parser.add_argument("runname", help="Name of run", type=str)
    return parser.parse_args()


def ara_loop(g):
    rundir = Path(f"rundata/{g.runname}")
    statefile = Path(rundir) / f"{g.runname}.yml"
    current_state = save_utils.get_current_state(statefile)

    if not rundir.exists():
        initialize.init(rundir)

    config_path = rundir / "config.yml"
    config.load_config(config_path)
    run_cfg = config.run()
    initialize.setup_logging(run_cfg["log_level"])

    for gen in range(current_state["generation"], run_cfg["total_gens"]):
        input(
            f"Starting generation {gen} at step {current_state['step']}. "
            "Press Enter to continue..."
        )
        # Generate DNA for generation

        # Build, simulate, export in XFdtd

        # Translate XF data into AraSim readable data, run AraSim, calculate fitness


if __name__ == "__main__":
    g = parse_args()
    ara_loop(g)
