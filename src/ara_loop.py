# GENETIS Main Loop
import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

from PyGA import Run_GA
from utils.settings import get, load_settings
from utils import initialize
from .xf.xfdtd_tools import XFRunner
import ara.arasim_tools as ara
from utils.save_state_utils import SaveState
import utils.plotting as plot


def parse_args():
    parser = argparse.ArgumentParser(description="Run Specific Details")
    parser.add_argument("run_name", help="Name of run", type=str)
    return parser.parse_args()


def ara_loop(g):
    workingdir = Path.cwd()
    run_dir = workingdir / "Run_Outputs" / g.run_name

    if not run_dir.exists():
        print("New run!")
        initialize.init(g.run_name)

    statefile = run_dir / f"{g.run_name}.yml"
    current_state = SaveState.load(SaveState, statefile)

    settings_path = run_dir / "settings.yaml"
    log_path = Path(run_dir / "job_outs/run.log")

    settings = load_settings(settings_path)

    initialize.setup_logging(get("log_level"), log_path)
    log = logging.getLogger(__name__)

    for gen in range(current_state.generation, get("n_gen")):
        if gen == 0:
            input(
                f"Starting generation {gen} at step: {current_state.step}. "
                "Press Enter to continue..."
            )
        else:
            print(f"Starting generation {gen} at step: {current_state.step}. ")

        if current_state.step == "ga":
            log.info("Creating generation directory")
            gen_dir = run_dir / "Generation_Data" / str(gen)
            gen_dir.mkdir(parents=True, exist_ok=True, mode=0o775)

            log.info("Generating genes...")
            Run_GA.main(
                args=SimpleNamespace(
                    run_name=g.run_name, workingdir=workingdir, gen=gen
                )
            )
            current_state.update("xf", gen, statefile)

        elif current_state.step == "xf":
            log.info("Simulating in XF...")
            xf = XFRunner(run_dir, g.run_name, gen, settings)
            xf.run_xf_step()
            current_state.update("ara", gen, statefile)

        elif current_state.step == "ara":
            log.info("Simulating in AraSim...")
            ara.run_ara_step()
            current_state.update("plot", gen, statefile)

        elif current_state.step == "plot":
            log.info("Plotting...")
            # plot()
            current_state.update("ga", gen, statefile)

        else:
            log.error(f"Unknown step '{current_state['step']}' found in savestate.")
            raise ValueError(f"Unknown step '{current_state['step']}' in savestate.")

    log.info(f"All {get("n_gen")} gens are done!!!")


if __name__ == "__main__":
    try:
        g = parse_args()
        ara_loop(g)

    except Exception as e:
        logging.getLogger(__name__).error(f"Unhandled exception: {e}", exc_info=True)
