from pathlib import Path
import subprocess
import logging
import shutil
import time

log = logging.getLogger(__name__)


class AnalyzeGen:
    def __init__(self, run_dir, gen, settings):
        self.run_dir = Path(run_dir)
        self.gen = gen
        self.settings = settings

        self.a_type = self.settings["a_type"]

        self.gen_dir = self.run_dir / "Generation_Data" / str(gen)
        self.txt_dir = self.gen_dir / "txt_files"
        self.csv_dir = self.gen_dir / "csv_files"

        self.top_individuals = []
        self.mid_individuals = []
        self.bottom_individuals = []
        self.best_overall = None

    def process_generation(self):
        self._load_fitness()
        self._select_top_mid_bottom()
        self._load_data()
        self._plot_all()
        self._update_overall_best()

    def _load_fitness(self):
        # load fitness data
        pass

    def _select_top_mid_bottom(self):
        # rank and store individuals
        pass

    def _load_data(self):
        # read vswr, s11, etc for selected individuals
        pass

    def _plot_single(self):
        # Plot single individual against Ara Antenna.

    def _plot_generation(self):
        # plot current generation 
        pass
    
    def _vswr_plot(self):
        # plot vswr data
        pass
    
    def _s11_plot(self):
        # plot s11 data
        pass
    
    def _imp_plot(self):
        # plot impedance data
        pass
    
    def _gain_plot(self):
        # plot gain data
        pass
    
    def _PoR_plot(self):
        # plot physics of results data
        pass

    def _update_overall_best(self):
        # compare to previous best
        pass
