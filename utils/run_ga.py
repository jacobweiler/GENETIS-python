# Functions related to running the GA portion of loop
from pathlib import Path
import argparse

from PyGA import Run_GA

def run_genetic_algorithm(runname: str, rundir: Path, gen: int):
    args = argparse.Namespace(run_name=runname, rundir=rundir, gen=gen)
    Run_GA.main(args)