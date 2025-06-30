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

        self.ara_dir = self.gen_dir / "ara_outputs"
        self.ara_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir = self.gen_dir / "csv_files"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.dat_dir = self.gen_dir / "dat_files"
        self.dat_dir.mkdir(parents=True, exist_ok=True)  
        

def _convert_uan_to_ara_format(self):
    # Antenna
    if self.a_type == "VPOL":
        gain_col = 2
        phase_col = 4
    elif self.a_type == "HPOL":
        gain_col = 3
        phase_col = 5
    else:
        raise ValueError(f"Unknown antenna type: {self.a_type}")

    freqVals = self.settings["freq_vals"]
    numFreq = len(freqVals)
    n = 37
    m = 73

    for indiv in range(1, self.npop + 1):
        dat_path = self.dat_dir / f"a_{indiv}.dat"
        with open(dat_path, "w") as datFile:
            for freq_idx, freq in enumerate(freqVals, start=1):
                datFile.write(f"freq : {freq:.2f} MHz\n")
                datFile.write("SWR : 1.965000\n")
                datFile.write(" Theta     Phi     Gain(dB)     Gain     Phase(deg) \n")

                uan_file = (
                    self.gen_dir
                    / "uan_files"
                    / f"{indiv}"
                    / f"{self.gen}_{indiv}_{freq_idx}.uan"
                )

                if not uan_file.exists():
                    raise FileNotFoundError(f"Missing UAN file: {uan_file}")

                try:
                    data = np.loadtxt(uan_file, skiprows=18, usecols=(0,1,2,3,4,5))
                except Exception as e:
                    raise ValueError(f"Error loading {uan_file}: {e}")

                theta = data[:, 0]
                phi   = data[:, 1]
                gain_db = data[:, gain_col]
                phase = data[:, phase_col]
                linear_gain = 10 ** (gain_db / 10)

                for t, p, gdb, lg, ph in zip(theta, phi, gain_db, linear_gain, phase):
                    datFile.write(
                        f"{t:.2f}\t{p:.2f}\t{gdb:.2f}\t{lg:.2f}\t{ph:.2f}\n"
                    )
        log.info(f"Converted UAN to Ara .dat format for individual {indiv} at {dat_path}")

    def _submit_arasim_jobs(self):
        total_jobs = self.npop * self.seeds
        cmd = [
            "sbatch",
            f"--array=1-{total_jobs}%{min(total_jobs, 50)}",
            f"--export=ALL,WorkingDir={self.working_dir},RunName={self.run_name},gen={self.gen},Seeds={self.seeds}",
            f"{self.working_dir}/Batch_Jobs/ara_job.sh"
        ]
        subprocess.run(cmd, check=True)
        log.info(f"Submitted AraSim job array for generation {self.gen}.")

    def _check_arasim_completion(self, poll_interval_sec=150):
        expected_count = self.npop * self.seeds * self.threads
        while True:
            txt_files = list(self.ara_dir.glob("AraOut_*.txt"))
            if len(txt_files) >= expected_count:
                log.info("All AraSim output files found.")
                break
            log.info(f"Waiting on AraSim outputs: {len(txt_files)}/{expected_count} found.")
            time.sleep(poll_interval_sec)

    def _calculate_fitness(self):
        """
        Calculate the AraSim fitness for each individual.
        """
        bhrad = 7.5 / self.settings.get("geoscalefactor", 1.0)
        scalefactor = self.settings.get("scalefactor", 1.0)
        
        # load DNA data
        dna_file = self.gen_dir / f"{self.gen}_generationDNA.csv"
        dna = np.loadtxt(dna_file, delimiter=",")

        fitnesses, veffs, fit_lowerrors, fit_higherrors = [], [], [], []
        lowerrors, higherrors = [], []

        for indiv in range(1, self.npop + 1):
            veff_sum = 0.0
            sqerr_low = 0.0
            sqerr_high = 0.0
            seeds_successful = self.seeds * self.threads

            values = []
            for seed in range(1, self.seeds + 1):
                for thread in range(self.threads):
                    idx = (seed - 1) * self.threads + thread
                    path = self.ara_dir / f"AraOut_{self.gen}_{indiv}_{idx}.txt"
                    try:
                        with open(path, "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                if "Veff(ice) :" in line:
                                    veff = float(line.split()[3])
                                if "error plus :" in line:
                                    parts = line.split(" : ")[1:]
                                    errplus = float(parts[0].split()[0])
                                    errminus = float(parts[1].split()[0])
                        if veff:
                            values.append(veff)
                            veff_sum += veff
                            sqerr_low += errplus**2
                            sqerr_high += errminus**2
                        else:
                            seeds_successful -= 1
                    except Exception:
                        seeds_successful -= 1
                        continue

            if seeds_successful == 0:
                avg_veff = 0.0
                avg_lowerr = 0.0
                avg_higherr = 0.0
                log.warning(f"No successful AraSim seeds for individual {indiv}")
            else:
                avg_veff = veff_sum / seeds_successful
                avg_lowerr = np.sqrt(sqerr_low) / seeds_successful
                avg_higherr = np.sqrt(sqerr_high) / seeds_successful

            indiv_data = dna[indiv - 1]
            if self.a_type == "VPOL":
                max_xy = indiv_data[0]
            elif self.a_type == "HPOL":
                max_xy = indiv_data[1] + 0.02

            penalty = 1.0
            if max_xy >= bhrad and self.settings.get("bh_penalty", 0):
                penalty = np.exp(-(scalefactor * (max_xy - bhrad) ** 2))

            fitness = avg_veff * penalty
            fitnesses.append(fitness)
            veffs.append(avg_veff)
            fit_lowerrors.append(avg_lowerr * penalty)
            fit_higherrors.append(avg_higherr * penalty)
            lowerrors.append(avg_lowerr)
            higherrors.append(avg_higherr)

            log.info(f"Individual {indiv}: Veff={avg_veff:.4e}, penalty={penalty:.4f}, fitness={fitness:.4e}")

        # save results
        np.savetxt(self.csv_dir / f"{self.gen}_Fitness.csv", fitnesses, delimiter=",")
        np.savetxt(self.csv_dir / f"{self.gen}_Fitness_Error.csv",
                   np.column_stack((fit_higherrors, fit_lowerrors)), delimiter=",")
        np.savetxt(self.csv_dir / f"{self.gen}_Veff.csv", veffs, delimiter=",")
        np.savetxt(self.csv_dir / f"{self.gen}_Veff_Error.csv",
                   np.column_stack((higherrors, lowerrors)), delimiter=",")

        log.info(f"AraSim fitness results saved to {self.csv_dir}")

    def _arasim_jobs_still_running(self):
        try:
            result = subprocess.run(
                ["squeue", "-n", self.run_name, "--noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                text=True,
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def _is_already_done(self):
        expected_output = self.gen_dir / "csv_files" / f"{self.gen}_Fitness.csv"
        return expected_output.exists()

    def run_ara_step(self):
        try:
            if self._is_already_done():
                log.info(f"Generation {self.gen}: AraSim step already completed.")
                return

            if self._arasim_jobs_still_running():
                log.info("AraSim jobs already running — skipping submission and waiting for completion.")
                self._check_arasim_completion()
                self._calculate_fitness()
            else:
                self._convert_uan_to_ara_format()
                self._submit_arasim_jobs()
                self._check_arasim_completion()
                self._calculate_fitness()

        except Exception as e:
            log.error(f"AraSim step failed: {e}", exc_info=True)

