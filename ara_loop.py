# GENETIS Main Loop
import argparse
import logging
from pathlib import Path

from utils import config, initialize
import src.xf.xfdtd_tools as xf
import src.ara.arasim_tools as ara
from utils.save_state_utils import SaveState
import utils.plotting as plot
import utils.run_ga as run_ga

def parse_args():
    parser = argparse.ArgumentParser(description="Run Specific Details")
    parser.add_argument("runname", help="Name of run", type=str)
    return parser.parse_args()


def ara_loop(g):
    rundir = Path(f"rundata/{g.runname}")
    statefile = Path(rundir) / f"{g.runname}.yml"
    current_state = SaveState.load(statefile)

    if not rundir.exists():
        initialize.init(rundir)

    config_path = rundir / "config.yml"
    log_path = Path(rundir / "job_outs/run.log")

    config.load_config(config_path)
    run_cfg = config.run()

    initialize.setup_logging(run_cfg["log_level"], log_path)
    log = logging.getLogger(__name__)

    for gen in range(current_state["generation"], run_cfg["total_gens"]):
        input(
            f"Starting generation {gen} at step {current_state['step']}. "
            "Press Enter to continue..."
        )
        
        if current_state["step"] == "genes":
            log.info("Generating genes...")
            run_ga(g.runname, rundir, gen)
            current_state.update("xf", gen, statefile)

        elif current_state["step"] == "xf": 
            log.info("Simulating in XF...")
            xf.run_xf_step()
            current_state.update("ara", gen, statefile)

        elif current_state["step"] == "ara":
            log.info("Simulating in AraSim...")
            ara.run_ara_step()
            current_state.update("plot", gen, statefile)

        elif current_state["step"] == "plot":
            log.info("Plotting...")
            plot()
            current_state.update("genes", gen, statefile)

        else:
            log.error(f"Unknown step '{current_state['step']}' found in savestate.")
            raise ValueError(f"Unknown step '{current_state['step']}' in savestate.")
        
    log.info(f"All {run_cfg["total_gens"]} gens are done!!!")



if __name__ == "__main__":
    try:
        g = parse_args()
        ara_loop(g)

    except Exception as e:
        logging.getLogger(__name__).error(f"Unhandled exception: {e}", exc_info=True)
