import logging
import shutil
import subprocess
import time
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# plt.switch_backend('agg')

log = logging.getLogger(__name__)


class AnalyzeGen:
    def __init__(self, run_dir, gen, settings):
        self.run_dir = Path(run_dir)
        self.gen = gen
        self.settings = settings

        self.workingdir = self.settings["workingdir"]
        self.npop = self.settings["npop"]
        self.ara_scripts = Path(self.settings["ara_scripts"])
        self.a_type = self.settings["a_type"]

        self.ara_vpol_dir = self.ara_scripts / "antenna_files" / "vpol"
        self.ara_hpol_dir = self.ara_scripts / "antenna_files" / "hpol"
        self.gen_dir = self.run_dir / "Generation_Data" / str(gen)
        self.plot_dir = self.run_dir / "plots"
        self.txt_dir = self.gen_dir / "txt_files"
        self.csv_dir = self.gen_dir / "csv_files"

        # Generation Data
        self.gen_dict = self._load_gen_data(self.gen)
        log.info("Loaded in generation dataframe")
        self.top_mid_worst_df = None

        # Plotting
        self.colors = [
            "#00429d",
            "#3e67ae",
            "#618fbf",
            "#85b7ce",
            "#b1dfdb",
            "#ffcab9",
            "#fd9291",
            "#e75d6f",
            "#c52a52",
            "#93003a",
        ]

    def process_gen(self):
        """
        Analyze current generation
        """
        self._top_mid_worst()
        self._update_overall_best()
        self._plot_gen()

    def _load_gen_data(self, gen):
        """
        Load generation data:
        - Generation
        - Fitness
        - Fitness Error
        - VSWR
        - S11
        - Impedance
        - DNA
        """
        gen_dict = {}
        gen_dir = self.run_dir / "Generation_Data" / str(gen)
        gen_fit_file = gen_dir / f"{gen}_fitnessScores.csv"
        gen_err_file = gen_dir / f"{gen}_fitnessError.csv"
        dna_file = gen_dir / f"{gen}_generationDNA.csv"

        # Load fitness and error data
        fit_scores = np.loadtxt(gen_fit_file, delimiter=",")
        fit_errors = np.loadtxt(gen_err_file, delimiter=",")
        dna_array = np.loadtxt(dna_file, delimiter=",")

        for i in range(1, self.npop + 1):
            indiv_file = gen_dir / "csv_files" / f"{gen}_{i}_vswr_s11_imp.csv"
            df = pd.read_csv(indiv_file)

            gen_dict[i] = {
                "Gen": gen,
                "VSWR": df["VSWR"].to_numpy(),
                "S11": df["| Reflection Coefficient |"].to_numpy(),
                "Impedance": df["| Z | (ohm)"].to_numpy(),
                "Fitness": fit_scores[i - 1],
                "FitnessError": fit_errors[i - 1],
                "DNA": dna_array[i - 1],  # 1D array
            }

        return gen_dict

    def _top_mid_worst(self):
        """
        Get generation top 3 best, middle 2, and worst individuals
        """

        # Ensure Fitness and FitnessError are scalars
        fitness = np.array(
            [np.atleast_1d(self.gen_dict[i]["Fitness"])[0] for i in self.gen_dict]
        )
        fit_error = np.array(
            [np.atleast_1d(self.gen_dict[i]["FitnessError"])[0] for i in self.gen_dict]
        )

        # Identify top 3, worst, and middle 2 individuals
        top_3_idx = np.argsort(fitness)[-3:][::-1]
        worst_idx = np.argmin(fitness)
        mean_fitness = np.mean(fitness)
        abs_diff = np.abs(fitness - mean_fitness)
        avg_2_idx = np.argsort(abs_diff)[:2]

        # Combine indices and remove duplicates
        all_idx = np.unique(np.concatenate([top_3_idx, avg_2_idx, [worst_idx]])).astype(
            int
        )

        # Generate labels ensuring length matches all_idx
        labels = [
            "top"
            if i in top_3_idx
            else "bottom"
            if i == worst_idx
            else "average"
            if i in avg_2_idx
            else "unknown"
            for i in all_idx
        ]

        # Debugging check
        print(
            "all_idx length:",
            len(all_idx),
            "labels length:",
            len(labels),
            "fitness shape:",
            fitness[all_idx].shape,
            "fit_error shape:",
            fit_error[all_idx].shape,
        )

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Gen": [self.gen] * len(all_idx),
                "Indiv": all_idx + 1,
                "Fitness": fitness[all_idx],
                "FitnessError": fit_error[all_idx],
                "label": labels,
            }
        )

        self.top_mid_worst_df = df
        df.to_csv(self.csv_dir / f"{self.gen}_top_mid_worst.csv", index=False)

    def _update_overall_best(self):
        """
        See if there is new best overall individual
        If new best => Make following Plots:
        1. Gain Comparison w/ Ara
        2. VSWR, S11, Imp w/ Ara (only have Ara data for VSWR + Imp)
        And save all relevant data
        """
        best_dir = self.run_dir / "best_overall"
        best_file = best_dir / "best_data.csv"

        gen_best_idx = self.top_mid_worst_df["Indiv"].iloc[0]
        gen_best_df = self.gen_dict[gen_best_idx]
        gen_best_df["Indiv"] = gen_best_idx

        if best_dir.exists():
            log.info("Loading in best overall dataframe...")
            best_df = pd.read_csv(best_file)
            if gen_best_df["Fitness"] > best_df["Fitness"].values[0]:
                log.info("New best found!!! Copying over data...")
                self._write_best_data(best_dir, gen_best_df)
        else:
            log.info("No best directory detected, creating and filling...")
            best_dir.mkdir(parents=True, exist_ok=True, mode=0o775)
            self._write_best_data(best_dir, gen_best_df)

    def _write_best_data(self, best_dir, best_data):
        uan_dir = self.gen_dir / "uan_files" / str(best_data["Indiv"])
        root_dir = self.gen_dir / "root_files" / str(best_data["Indiv"])

        best_uan_dir = best_dir / "uan_files"
        best_root_dir = best_dir / "root_files"
        best_plot_dir = best_dir / "plots"

        # Clearing best directories
        if best_dir.exists():
            shutil.rmtree(best_dir)
        best_dir.mkdir(parents=True, exist_ok=True, mode=0o775)
        best_uan_dir.mkdir(parents=True, exist_ok=True, mode=0o775)
        best_root_dir.mkdir(parents=True, exist_ok=True, mode=0o775)
        best_plot_dir.mkdir(parents=True, exist_ok=True, mode=0o775)

        # copying over new best files
        self._copy_all_files(uan_dir, best_uan_dir)
        self._copy_all_files(root_dir, best_root_dir)

        # save new best data
        pd.DataFrame([best_data]).to_csv(best_dir / "best_data.csv", index=False)

        # Plotting
        self._gain_plot([best_uan_dir], ["Best Overall"], best_plot_dir)
        self._gain_freq_plot([best_uan_dir], ["Best Overall"], best_plot_dir)
        self._vswr_s11_imp_plot(
            [
                best_dir
                / "csv_files"
                / f"{self.gen}_{best_data['Indiv']}_vswr_s11_imp.csv"
            ],
            ["Best Overall"],
            best_plot_dir,
        )

    def _copy_all_files(self, src_dir, dest_dir):
        """
        copies all files in src_dir to dest_dir
        """
        src_dir = Path(src_dir)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for file in src_dir.iterdir():
            if file.is_file():
                shutil.copy2(file, dest_dir / file.name)

    def _plot_gen(self):
        """
        Plots the following:
        1. Update Fitness plot to include new gen
        2. Plot the gain of Top 3, middle 2, and worst individuals
        3. Plot VSWR, S11, Imp for those individuals
        """
        summary = []
        indivs_to_plot = []
        labels = []

        # Get fitness stats over generations
        for gen in range(self.gen + 1):
            gen_dir = self.run_dir / "Generation_Data" / str(gen)
            fit_file = gen_dir / f"{gen}_fitnessScores.csv"

            if not fit_file.exists():
                continue

            try:
                fit_scores = np.loadtxt(fit_file, delimiter=",")
            except Exception as e:
                log.warning(f"Could not load fitness for gen {gen}: {e}")
                continue

            summary.append(
                {
                    "Generation": gen,
                    "Mean": np.mean(fit_scores),
                    "Median": np.median(fit_scores),
                    "Min": np.min(fit_scores),
                    "Max": np.max(fit_scores),
                    "Std": np.std(fit_scores),
                }
            )

        gens_fit_df = pd.DataFrame(summary)

        # Plotting
        self._fitness_plot(gens_fit_df, self.plot_dir)

        # Top, mid and worst individuals
        for _, row in self.top_mid_worst_df.iterrows():
            indivs_to_plot.append(int(row["Indiv"]))
            labels.append(row["label"])
        gain_dirs = [
            self.gen_dir / "uan_files" / str(indiv) / f"{self.gen}_{indiv}_"
            for indiv in indivs_to_plot
        ]
        vswr_s11_imp_dirs = [
            self.csv_dir / f"{self.gen}_{indiv}_vswr_s11_imp.csv"
            for indiv in indivs_to_plot
        ]

        self._gain_plot(gain_dirs, labels, self.plot_dir)
        self._vswr_s11_imp_plot(vswr_s11_imp_dirs, labels, self.plot_dir)

    def _fitness_plot(self, gen_data, outloc):
        # plot generation fitness alongside other generations
        generations = gen_data["Generation"]
        mean_vals = gen_data["Mean"]
        median_vals = gen_data["Median"]
        min_vals = gen_data["Min"]
        max_vals = gen_data["Max"]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot mean with error as vertical line (min/max)
        ax.plot(generations, mean_vals, color="darkorange", label="Mean", linewidth=2)
        ax.fill_between(
            generations,
            mean_vals - gen_data["Std"],
            mean_vals + gen_data["Std"],
            color="orange",
            alpha=0.2,
            label="Std Dev",
        )

        # Plot vertical error bars
        for i in range(len(generations)):
            ax.vlines(
                generations[i], min_vals[i], max_vals[i], color="black", alpha=0.2
            )

        # Plot median
        ax.plot(
            generations,
            median_vals,
            linestyle="--",
            color="teal",
            linewidth=2,
            label="Median",
        )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness Score (V_eff [km³ sr])")
        ax.set_title(f"Fitness Score over Generations (0 - {generations.max()})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(outloc / f"{self.gen}_Fitness.png")
        plt.savefig(outloc / f"{self.gen}_Fitness.pdf")
        plt.close()

    def _gain_plot(self, uan_dirs, labels, outloc):
        """
        Input list of uan dir and labels and output gain plots vs Ara antenna
        For each inputted uan dir outputs plots from 100 MHz to 800 MHz
        """
        freq_list = [
            (i, freq) for i, freq in zip(range(2, 51, 6), range(100, 801, 100))
        ]
        if self.a_type == "VPOL":  # Need to put the target files into directories
            # theta_slice = 90  # Put theta slice here that we want for VPOL
            ara_gain_dir = self.ara_vpol_dir / "ara_bicone6in_"
            # ara_label = "Ara VPOL"
        elif self.a_type == "HPOL":
            # theta_slice = 90  # Put theta slice wanted for HPOL
            ara_gain_dir = (
                self.ara_hpol_dir / "ara_hpol_"
            )  # Need to get 100 MHz + 800 MHz files
            # ara_label = "Ara HPOL"

        for i, directory in enumerate(uan_dirs):
            plot_out_dir = outloc / labels[i]
            plot_out_dir.mkdir(parents=True, exist_ok=True, mode=0o775)
            for freq in freq_list:
                title = f"{freq[1]} MHz"
                curr_label = labels[i]
                out_name = f"{curr_label}_gain_{round(freq[1])}"

                # Ara Antenna
                ara_file = ara_gain_dir.with_name(ara_gain_dir.stem + f"{freq[0]}.csv")
                ara_data = np.genfromtxt(
                    ara_file, names=True, dtype=None, encoding=None, skip_header=2
                )
                # ara_theta = ara_data['Theta']
                ara_phi = ara_data["Phi"]
                ara_gain = ara_data["GaindB"]
                gain_ara_0 = ara_gain[ara_phi == 0]
                gain_ara_180 = ara_gain[ara_phi == 180]
                th_gain0 = np.concatenate([gain_ara_0, gain_ara_180])
                # not sure if this is needed
                # phi_gain0 = ara_gain[ara_theta == 90]

                # GENETIS Antenna
                uan_file = directory.with_name(directory.stem + f"{freq[0]}.uan")
                uan_data = np.genfromtxt(uan_file, unpack=True, skip_header=18)
                # thetas = np.radians(uan_data[0])
                phis = np.radians(uan_data[1])
                uan_gain_th = uan_data[2]  # Theta polarized gain
                # Get gain at 0 and 180 degrees in theta slice
                # This is getting the theta (vertically) polarized gain
                gain1_0 = uan_gain_th[phis == np.radians(0)]
                gain1_180 = uan_gain_th[phis == np.radians(180)]
                th_gain1 = np.concatenate([gain1_0, gain1_180])
                # Phi slices of gain at theta = 90
                # phi_gain1 = uan_gain_th[thetas = np.radians(90)]

                # Setting up for Zenith (theta), idk if azimuth is wanted
                zenith_ang = phis[1314:1387]

                # Setting up plots (doing it in a way that azimuth could easily be added)
                fig = plt.figure(figsize=(10, 6))
                ax1 = fig.add_subplot(1, 2, 1, polar=True)

                # Style
                mpl.rcParams["text.usetex"] = True
                plt.style.use("seaborn-white")

                # Plot settings
                ax1.set_aspect("equal")
                all_gains = np.concatenate([th_gain0, th_gain1])
                min_gain = -40
                max_gain = np.ceil(all_gains.max()) + 3
                ax1.set_theta_zero_location("N")
                ax1.set_rlabel_position(225)
                ax1.tick_params(axis="both", which="major", labelsize=11.5, pad=7)
                ax1.set_xticklabels(
                    ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"],
                    fontsize=14,
                )
                ax1.set_ylim(min_gain, max_gain)
                ax1.set_yticks(np.arange(min_gain, max_gain, 10))
                ax1.set_yticklabels(
                    [str(int(t)) for t in np.arange(min_gain, max_gain, 10)],
                    fontsize=14,
                )
                ax1.text(
                    0.5,
                    1.11,
                    "Zenith",
                    fontsize=18,
                    ha="center",
                    transform=ax1.transAxes,
                )

                # Add Data to Plot
                ax1.plot(
                    zenith_ang,
                    th_gain0[:-1],
                    color=self.colors[0],
                    lw=3.5,
                    linestyle="dotted",
                    alpha=1,
                )
                ax1.plot(
                    zenith_ang,
                    th_gain1[:-1],
                    color=self.colors[1],
                    lw=3,
                    linestyle="dashed",
                    alpha=1,
                )

                # Output Plot
                title_size = 40 * (33.87 / 39.85)
                plt.suptitle(title, fontsize=title_size, y=0.95)
                plt.savefig(outloc / out_name, dpi=300, bbox_inches="tight")
                plt.close(fig)

    def _vswr_s11_imp_plot(self, data_dirs, labels, outloc):
        """
        Plots vswr, s11, and impedance for each inputted data_dir
        """
        # file_tail = "vswr_s11_imp.csv"

        for i, directory in enumerate(data_dirs):
            # The directory in data_dirs isn't the full file path. CHANGE THIS.
            data = np.genfromtxt(directory, delimiter=",", skip_header=1)
            # VSWR
            plt.figure(figsize=(16, 8))
            plt.plot(
                data[:, 0] / (10**9),
                data[:, 1],
                label=labels[i],
                color=self.colors[0],
                linestyle="solid",
                linewidth=3,
            )
            plt.ylim(0.8, 5)
            # plt.xlim(0.1, .8)
            plt.xlabel("Frequency (GHz)", fontsize=18)
            plt.ylabel("VSWR", fontsize=18)
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            # plt.title("VSWR", fontsize=18)
            plt.savefig(outloc / f"{self.gen}_{labels[i]}_vswr.png")

            # S11
            plt.figure(figsize=(16, 8))
            plt.plot(
                data[:, 0] / (10**9),
                20 * np.log10(data[:, 2]),
                label=labels[i],
                color=self.colors[0],
                linestyle="solid",
                linewidth=3,
            )
            # plt.ylim(-35, 0)
            # plt.xlim(.2, .8)
            plt.xlabel("Frequency (GHz)", fontsize=18)
            plt.ylabel("|S11|", fontsize=18)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(outloc / f"{self.gen}_{labels[i]}_s11.png")

            # Impedance
            plt.figure(figsize=(16, 8))
            plt.plot(
                data[:, 0] / (10**9),
                data[:, 3],
                label=labels[i],
                color=self.colors[0],
                linestyle="solid",
                linewidth=3,
            )
            # plt.ylim(-35, 0)
            # plt.xlim(.2, .8)
            plt.xlabel("Frequency (GHz)", fontsize=18)
            plt.ylabel("Impedance (Ohm)", fontsize=18)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(outloc / f"{self.gen}_{labels[i]}_imp.png")
