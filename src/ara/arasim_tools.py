import os
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

        self.workingdir = Path(settings["workingdir"])
        self.npop = settings["npop"]
        self.processes = settings["ara_processes"]
        self.ara_exec = Path(settings["ara_exec"])
        self.a_type = settings["a_type"]
        self.threads = settings["job_threads"]
        self.nnt = settings["nnt"]
        self.exp = settings["exp"]

        self.ara_dir = self.gen_dir / "ara_outputs"
        self.ara_dir.mkdir(parents=True, exist_ok=True)
        self.txt_dir = self.gen_dir / "txt_files"
        self.txt_dir.mkdir(parents=True, exist_ok=True)

    def _convert_uan_to_ara_format(self):
        freq_vals = self.settings["freq_vals"]

        n_theta = 37
        n_phi = 73
        numFreq = 60

        head1_a = "freq : "
        head1_b = " MHz"
        head2 = "SWR : 1.965000"
        head3 = " Theta     Phi     Gain(dB)     Gain     Phase(deg) "

        # choose columns depending on antenna type
        if self.a_type == "VPOL":
            gain_col = 2
            phase_col = 4
        elif self.a_type == "HPOL":
            gain_col = 3
            phase_col = 5
        else:
            raise ValueError(f"Unknown antenna type: {self.a_type}")

        for antenna in range(self.npop):
            txt_path = self.txt_dir / f"a_{antenna + 1}.txt"
            with open(txt_path, "w") as txtFile:
                os.chmod(txt_path, 0o777)
                for freq_idx in range(numFreq):
                    txtFile.write(head1_a + str(freq_vals[freq_idx]) + head1_b + "\n")
                    txtFile.write(head2 + "\n")
                    txtFile.write(head3 + "\n")

                    uan_file = (
                        self.gen_dir
                        / "uan_files"
                        / f"{antenna + 1}"
                        / f"{self.gen}_{antenna + 1}_{freq_idx + 1}.uan"
                    )

                    if not uan_file.exists():
                        raise FileNotFoundError(f"Missing UAN file: {uan_file}")

                    with open(uan_file, "r") as f:
                        # skip headers
                        for _ in range(18):
                            f.readline()

                        mat = [["0" for _ in range(n_theta)] for _ in range(n_phi)]
                        for i in range(n_theta):
                            for j in range(n_phi):
                                line = f.readline()
                                parts = line.split()
                                theta_val = parts[0]
                                phi_val = parts[1]
                                gain_db = float(parts[gain_col])
                                phase_deg = float(parts[phase_col])
                                linear_gain = 10 ** (gain_db / 10)

                                lineFinal = (
                                    f"{theta_val} \t "
                                    f"{phi_val} \t "
                                    f"{gain_db:.2f}     \t   "
                                    f"{linear_gain:.2f}     \t    "
                                    f"{phase_deg:.2f}\n"
                                )

                                mat[j][i] = lineFinal

                        for p in range(n_phi - 1):
                            for q in range(n_theta):
                                txtFile.write(mat[p][q])

            log.info(
                f"UAN to Ara .txt conversion done for antenna {antenna + 1} at {txt_path}"
            )

    def _submit_arasim_jobs(self):
        total_jobs = self.npop * self.processes
        nnt_per_ara = round(self.nnt / self.processes)
        cmd = [
            "sbatch",
            f"--array=1-{total_jobs}%{min(total_jobs, 50)}",
            f"--export=ALL,WorkingDir={self.workingdir},RunName={self.run_name},"
            f"gen={self.gen},Seeds={self.processes},threads={self.settings['job_threads']},"
            f"AraSimDir={self.settings['ara_exec']},a_type={self.a_type},"
            f"SpecificSeed={self.settings['rng_seed']},RunDir={self.run_dir},"
            f"nnt_per_ara={nnt_per_ara},exp={self.exp}",
            f"--output={self.run_dir}/job_outs/ara_out/AraSim_%a.output",
            f"--error={self.run_dir}/job_outs/ara_err/AraSim_%a.error",
            f"{self.workingdir}/src/ara/ara_job.sh",
        ]
        subprocess.run(cmd, check=True)
        log.info(f"Submitted AraSim job array for generation {self.gen}.")

    def _check_arasim_completion(self, poll_interval_sec=150):
        expected_count = self.npop * self.processes * self.threads
        while True:
            txt_files = list(self.ara_dir.glob("AraOut_*.txt"))
            if len(txt_files) >= expected_count:
                log.info("All AraSim output files found.")
                break
            log.info(
                f"Waiting on AraSim outputs: {len(txt_files)}/{expected_count} found."
            )
            time.sleep(poll_interval_sec)

    def _calculate_fitness(self):
        bhrad = 7.5 / self.settings["geo_factor"]
        scalefactor = self.settings["scale_factor"]

        dna_file = self.gen_dir / f"{self.gen}_generationDNA.csv"
        dna = np.loadtxt(dna_file, delimiter=",")

        fitnesses, veffs, fit_lowerrors, fit_higherrors = [], [], [], []
        lowerrors, higherrors = [], []

        for indiv in range(1, self.npop + 1):
            veff_sum = 0.0
            sqerr_low = 0.0
            sqerr_high = 0.0
            processes_successful = self.processes * self.threads

            values = []
            for process in range(1, self.processes + 1):
                for thread in range(self.threads):
                    idx = (process - 1) * self.threads + thread
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
                        if veff is not None:
                            values.append(veff)
                            veff_sum += veff
                            sqerr_low += errplus**2
                            sqerr_high += errminus**2
                        else:
                            processes_successful -= 1
                    except Exception:
                        processes_successful -= 1
                        continue

            if processes_successful == 0:
                avg_veff = 0.0
                avg_lowerr = 0.0
                avg_higherr = 0.0
                log.warning(f"No successful AraSim processes for individual {indiv}")
            else:
                avg_veff = veff_sum / processes_successful
                avg_lowerr = np.sqrt(sqerr_low) / processes_successful
                avg_higherr = np.sqrt(sqerr_high) / processes_successful

            indiv_data = dna[indiv - 1]
            if self.a_type == "VPOL":
                max_xy = indiv_data[0]
            elif self.a_type == "HPOL":
                max_xy = indiv_data[1] + 0.02

            penalty = 1.0
            if max_xy >= bhrad:
                penalty = np.exp(-(scalefactor * (max_xy - bhrad) ** 2))

            fitness = avg_veff * penalty
            fitnesses.append(fitness)
            veffs.append(avg_veff)
            fit_lowerrors.append(avg_lowerr * penalty)
            fit_higherrors.append(avg_higherr * penalty)
            lowerrors.append(avg_lowerr)
            higherrors.append(avg_higherr)

            log.info(
                f"Individual {indiv}: Veff={avg_veff:.4e}, penalty={penalty:.4f}, "
                f"fitness={fitness:.4e}"
            )

        np.savetxt(self.gen_dir / f"{self.gen}_fitnessScores.csv", fitnesses, delimiter=",")
        np.savetxt(
            self.gen_dir / f"{self.gen}_Fitness_Error.csv",
            np.column_stack((fit_higherrors, fit_lowerrors)),
            delimiter=",",
        )
        np.savetxt(self.gen_dir / f"{self.gen}_Veff.csv", veffs, delimiter=",")
        np.savetxt(
            self.gen_dir / f"{self.gen}_Veff_Error.csv",
            np.column_stack((higherrors, lowerrors)),
            delimiter=",",
        )

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
        expected_output = self.gen_dir / f"{self.gen}_fitnessScores.csv"
        return expected_output.exists()

    def _arasim_outputs_exist(self):
        return any(self.ara_dir.glob("AraOut_*.txt"))

    def run_ara_step(self):
        try:
            if self._is_already_done():
                log.info(f"Generation {self.gen}: AraSim step already completed.")
                return
            if self._arasim_jobs_still_running():
                log.info("AraSim jobs already running — waiting for completion.")
                self._check_arasim_completion()
                self._calculate_fitness()
            elif self._arasim_outputs_exist():
                log.info(
                    "AraSim outputs exist and no jobs running"
                    "— proceeding to calculate fitness."
                )
                self._calculate_fitness()
            else:
                self._convert_uan_to_ara_format()
                self._submit_arasim_jobs()
                self._check_arasim_completion()
                self._calculate_fitness()

        except Exception as e:
            log.error(f"AraSim step failed: {e}", exc_info=True)
