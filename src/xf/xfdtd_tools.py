import sys
from pathlib import Path
import subprocess
import logging
import shutil
import time

log = logging.getLogger(__name__)


class XFRunner:
    def __init__(self, run_dir: Path, run_name: str, gen: int, settings: dict):
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.gen = gen
        self.settings = settings

        self.working_dir = Path(settings["workingdir"])
        self.npop = settings["npop"]
        self.xf_proj = Path(settings["xf_proj"])
        self.xmacros_dir = Path(settings["xmacros"])
        self.run_xmacros_dir = Path(settings["run_xmacros"])
        self.a_type = settings["a_type"]

    def _get_sim_num(self, indiv_index: int):
        return self.gen * self.npop + indiv_index

    def _get_status_file(self, sim_num: int):
        return (
            self.run_dir
            / f"{self.run_name}.xf"
            / "Simulations"
            / f"{sim_num:06d}"
            / "Run0001"
            / "output"
            / "status"
            / "runstatus.complete"
        )

    def _all_simulations_done(self):
        return all(
            self._get_status_file(self._get_sim_num(i)).exists()
            for i in range(1, self.npop + 1)
        )

    def _clean_sim_dirs(self):
        for i in range(1, self.npop + 1):
            sim_dir = self.xf_proj / "Simulations" / f"{self._get_sim_num(i):06d}"
            if sim_dir.exists():
                shutil.rmtree(sim_dir)
        log.info("Old simulation directories cleaned.")

    def _create_simulation_macro_vpol(self):
        macro_path = self.run_xmacros_dir / "simulation_PEC.xmacro"
        with macro_path.open("w") as f:
            f.write(f"var NPOP = {self.npop};\n")
            f.write("var indiv = 1;\n")
            f.write(f"var gen = {self.gen};\n")
            f.write(f'var workingdir = "{self.working_dir}";\n')
            f.write(f'var RunName = "{self.run_name}";\n')
            f.write(f"var freq_start = {self.settings['freq_start']};\n")
            f.write(f"var freq_step = {self.settings['freq_step']};\n")
            f.write(f"var freqCoefficients = {self.settings['freq_num']};\n")
            f.write(f"var CURVED = {self.settings['curved']};\n")
            f.write(f"var NSECTIONS = {self.settings['nsections']};\n")
            f.write(f"var evolve_sep = {self.settings['seperation']};\n")
            if self.gen == 0:
                f.write(f'App.saveCurrentProjectAs("{self.run_dir/self.run_name}");\n')
        macro_parts = [
            "headerVPOL.js", "functioncallsVPOL.js", "build_vpol.js",
            "CreatePEC.js", "CreateVPOLAntennaSource.js", "CreateGridVPOL.js",
            "CreateSensors.js", "CreateAntennaSimulationData.js",
            "QueueSimulation.js", "MakeImage.js"
        ]
        for part in macro_parts:
            with open(self.xmacros_dir / part) as src:
                f.write(src.read())
        log.info(f"simulation_PEC.xmacro created at {macro_path}")

    def _create_simulation_macro_hpol(self):
        macro_path = self.run_xmacros_dir / "simulation_PEC.xmacro"
        with macro_path.open("w") as f:
            f.write(f"var NPOP = {self.npop};\n")
            f.write("var indiv = 1;\n")
            f.write(f"var gen = {self.gen};\n")
            f.write(f'var workingdir = "{self.working_dir}";\n')
            f.write(f'var RunName = "{self.run_name}";\n')
            f.write(f"var freq_start = {self.settings['freq_start']};\n")
            f.write(f"var freq_step = {self.settings['freq_step']};\n")
            f.write(f"var freqCoefficients = {self.settings['freq_num']};\n")
            if self.gen == 0:
                f.write(f'App.saveCurrentProjectAs("{self.run_dir/self.run_name}");\n')
        macro_parts = [
            "headerHPOL.js", "functioncallsHPOL.js", "build_hpol.js",
            "CreatePEC.js", "CreateHPOLAntennaSource.js", "CreateGridHPOL.js",
            "CreateSensors.js", "CreateAntennaSimulationData.js",
            "QueueSimulation.js", "MakeImage.js"
        ]
        for part in macro_parts:
            with open(self.xmacros_dir / part) as src:
                f.write(src.read())
        log.info(f"simulation_PEC.xmacro created at {macro_path}")

    def _run_build_macro(self):
        macro_path = self.run_xmacros_dir / "simulation_PEC.xmacro"
        cmd = ["xfdtd", str(self.xf_proj), f"--execute-macro-script={macro_path}"]
        try:
            subprocess.run(cmd, check=True)
            log.info("XF building macro executed.")
        except subprocess.CalledProcessError as e:
            log.warning("XF macro building execution failed or was interrupted.")
            log.warning(e)

    def _submit_jobs(self):
        num_keys = self.settings["num_keys"]
        batch_size = min(self.npop, num_keys)
        job_time = "04:00:00"
        subprocess.run(["scancel", "-n", self.run_name], check=False)
        cmd = [
            "sbatch",
            f"--array=1-{self.npop}%{batch_size}",
            f"--export=ALL,WorkingDir={self.working_dir},RunName={self.run_name},"
            f"indiv=0,gen={self.gen},batch_size={batch_size}",
            f"--job-name={self.run_name}",
            f"--time={job_time}",
            str(self.working_dir / "Batch_Jobs" / "GPU_XF_Job.sh"),
        ]
        subprocess.run(cmd, check=True)
        log.info(f"Submitted XF jobs with batch size {batch_size}.")

    def _create_output_macro(self):
        macro_path = self.run_xmacros_dir / "output.xmacro"
        with macro_path.open("w") as f:
            f.write(f"var popsize = {self.npop};\n")
            f.write(f"var gen = {self.gen};\n")
            f.write(f'var workingdir = "{self.working_dir}";\n')
            f.write(f'var RunDir = "{self.run_dir}";\n')
            with open(self.xmacros_dir / "output_skele.js") as src:
                f.write(src.read())
        log.info(f"output.xmacro created at {macro_path}")

    def _run_output_macro(self):
        macro_path = self.run_xmacros_dir / "output.xmacro"
        cmd = ["xfdtd", str(self.xf_proj), f"--execute-macro-script={macro_path}"]
        try:
            subprocess.run(cmd, check=True)
            log.info("XF output macro executed.")
        except subprocess.CalledProcessError as e:
            log.warning("XF output macro execution failed or was interrupted.")
            log.warning(e)

    def run_xf_step(self, poll_interval=150):
        """
        Run XF simulation step:
        - If simulations already complete, run output
        - If not complete but jobs are running, wait
        - If not complete and no jobs are running, submit jobs
        """
        log.info(f"Generation {self.gen}: Starting XF simulation step...")

        def jobs_still_running():
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

        if self._all_simulations_done():
            log.info(f"Generation {self.gen}: Simulations already completed. Skipping job submission.")
        elif jobs_still_running():
            log.info(f"Jobs for {self.run_name} are already running. Skipping build macro and job submission.")
        else:
            self._clean_sim_dirs()
            uan_dir = self.run_dir / "Generation_Data" / self.gen / "uan_files"
            uan_dir.mkdir(parents=True, exist_ok=True, mode=0o775)
            
            if self.a_type == "VPOL":
                self._create_simulation_macro_vpol()
            elif self.a_type == "HPOL":
                self._create_simulation_macro_hpol()
            else:
                raise ValueError(f"Unsupported antenna type: {self.a_type}")

            self._run_build_macro()
            self._submit_jobs()

        # Wait for simulations to complete
        if not self._all_simulations_done():
            log.info("Waiting for all simulations to complete...")
            while not self._all_simulations_done():
                log.info("Simulations not finished yet. Sleeping 2m30s...")
                time.sleep(poll_interval)
            log.info("All simulations complete.")

        self._create_output_macro()
        self._run_output_macro()
        log.info(f"Generation {self.gen}: Outputted UAN files.")
