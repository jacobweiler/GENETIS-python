from pathlib import Path
import subprocess
import logging
import time
import numpy as np

log = logging.getLogger(__name__)

class AraRunner:
    def __init__(self, run_dir: Path, run_name: str, gen: int, settings: dict):
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.gen = gen
        self.gen_dir = run_dir / "Generation_Data" / str(gen)
        self.settings = settings

        self.working_dir = Path(settings["workingdir"])
        self.npop = settings["npop"]
        self.seeds = settings["seeds"]
        self.ara_exec = Path(settings["arasim_exec"])
        self.ara_dir = Path(settings["arasim_dir"])
        self.a_type = settings["a_type"]
        self.threads = settings["job_threads"]

        self.ara_output_dir = self.run_dir / "AraSim_Outputs" / f"{gen}_AraSim_Outputs"
        self.ara_output_dir.mkdir(parents=True, exist_ok=True)

    def _convert_uan_to_ara_format(self):
        dat_out_dir = self.gen_dir / "dat_files"
        dat_out_dir.mkdir(parents=True, exist_ok=True)

        for indiv in range(1, self.npop + 1):
            uan_path = self.gen_dir / "uan_files" / f"{indiv}"
            a_dat = dat_out_dir / f"a_{indiv}.dat"
            gain_data = []
            for freq in range(1, self.settings["freq_num"] + 1):
                file = uan_path / f"{self.gen}_{indiv}_{freq}.uan"
                if not file.exists():
                    raise FileNotFoundError(f"Missing UAN file: {file}")
                data = np.loadtxt(file)
                gain_data.append(data[:, :2]) 

            gain_data = np.stack(gain_data, axis=0)
            averaged = np.mean(gain_data, axis=0)
            np.savetxt(a_dat, averaged)
            log.debug(f"Converted UAN to Ara format: {a_dat}")

    def _submit_arasim_jobs(self):
        total_jobs = self.npop * self.seeds
        cmd = [
            "sbatch",
            f"--array=1-{total_jobs}%{min(total_jobs, 100)}",
            f"--export=ALL,WorkingDir={self.working_dir},RunName={self.run_name},gen={self.gen},Seeds={self.seeds}",
            f"{self.working_dir}/Batch_Jobs/ara_job.sh"
        ]
        subprocess.run(cmd, check=True)
        log.info(f"Submitted AraSim job array for generation {self.gen}.")

    def _check_arasim_completion(self, poll_interval_sec=150):
        expected_count = self.npop * self.seeds * self.threads
        while True:
            txt_files = list(self.ara_output_dir.glob("AraOut_*.txt"))
            if len(txt_files) >= expected_count:
                log.info("All AraSim output files found.")
                break
            log.info(f"Waiting on AraSim outputs: {len(txt_files)}/{expected_count} found.")
            time.sleep(poll_interval_sec)

    def _calculate_fitness(self):
        gen_dna_file = self.run_dir / f"Generation_Data/{self.gen}/{self.gen}_generationDNA.csv"
        dna = np.loadtxt(gen_dna_file, delimiter=",")
        bhrad = 7.5 / self.settings.get("geoscalefactor", 1.0)
        scalefactor = self.settings.get("scalefactor", 1.0)
        a_type = self.a_type

        fitnesses, veffs = [], []
        for i in range(1, self.npop + 1):
            values = []
            for seed in range(1, self.seeds + 1):
                for thread in range(self.threads):
                    path = self.ara_output_dir / f"AraOut_{self.gen}_{i}_{(seed - 1)*self.threads + thread}.txt"
                    try:
                        with open(path, "r") as f:
                            val = float(f.readline().strip())
                            values.append(val)
                    except Exception:
                        continue

            avg_val = np.mean(values) if values else 0.0
            veffs.append(avg_val)

            # Determine max radius based on antenna type
            indiv_data = dna[i - 1]
            if a_type in [0, 1, 2]:
                max_xy = indiv_data[0]
            else:
                max_xy = indiv_data[1] + 0.02

            penalty = np.exp(-(scalefactor * (max_xy - bhrad) ** 2)) if max_xy >= bhrad else 1.0
            fitnesses.append(avg_val * penalty)

        # Save to CSV
        csv_dir = self.run_dir / "csv_files"
        csv_dir.mkdir(exist_ok=True)
        np.savetxt(csv_dir / f"{self.gen}_Fitness.csv", fitnesses, delimiter=",")
        log.info(f"Saved AraSim fitness to {csv_dir}/{self.gen}_Fitness.csv")

    def run_ara_step(self):
        try:
            if not self._is_already_done():
                self._convert_uan_to_ara_format()
                self._submit_arasim_jobs()
                self._check_arasim_completion()
                self._calculate_fitness()
            else:
                log.info(f"Generation {self.gen}: AraSim step already completed.")
        except Exception as e:
            log.error(f"AraSim step failed: {e}", exc_info=True)

    def _is_already_done(self):
        expected_output = self.run_dir / "csv_files" / f"{self.gen}_Fitness.csv"
        return expected_output.exists()
