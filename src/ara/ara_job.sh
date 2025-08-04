#!/bin/bash
## This job is designed to be submitted by an array batch submission
## Here's the command:
## sbatch --array=1-NPOP*SEEDS%max --export=ALL,(variables) ara_job.sh
#SBATCH -A PAS1960
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 40

source /fs/ess/PAS1960/BiconeEvolutionOSC/new_root/new_root_setup.sh
source /rhel7/etc/cvmfs/ara.opensciencegrid.org/trunk/centos7/setup.sh

cd $AraSimDir

outputloc="$RunDir/job_outs/ara_out/AraSim_${SLURM_ARRAY_TASK_ID}.output"
errorloc="$RunDir/job_outs/ara_err/AraSim_${SLURM_ARRAY_TASK_ID}.error"
num=$(($((${SLURM_ARRAY_TASK_ID}-1))/${Seeds}+1))
seed=$(($((${SLURM_ARRAY_TASK_ID}-1))%${Seeds}+1))
echo "num is $num"
echo "seed is $seed"
# init = seed - 1
init=$((${seed}-1))
echo "init is $init"

# creating setup file for current individual
gain_file="${RunDir}/Generation_Data/${gen}/txt_files/a_${num}.txt "
echo "gain file is $gain_file"

if [ "$a_type" = "VPOL" ]; then
    sed -e "s|num_nnu|$nnt_per_ara|" \
        -e "s|n_exp|$exp|" \
        -e "s|current_seed|$SpecificSeed|" \
        -e "s|vpol_gain|$gain_file|" \
        "${WorkingDir}/src/ara/setup_dummy_vpol.txt" > "$TMPDIR/setup.txt"
elif [ "$a_type" = "HPOL" ]; then
    sed -e "s|num_nnu|$nnt_per_ara|" \
        -e "s|n_exp|$exp|" \
        -e "s|current_seed|$SpecificSeed|" \
        -e "s|hpol_gain|$gain_file|" \
        "${WorkingDir}/src/ara/setup_dummy_hpol.txt" > "$TMPDIR/setup.txt"
else
    echo "Error: unknown a_type '$a_type'"
    exit 1
fi

# starts running $threads processes of AraSim
echo "Starting AraSim processes"
for (( i=0; i<${threads}; i++ ))
do
    # we need $threads unique id's for each seed
    indiv_thread=$((${init}*${threads}+${i}))
    dataoutloc="$TMPDIR/AraOut_${gen}_${num}_${indiv_thread}.txt"
    echo "individual thread is $indiv_thread"
    ./AraSim $TMPDIR/setup.txt ${indiv_thread} $TMPDIR > $dataoutloc &
done

simulationcheck=False
while [ "$simulationcheck" = False ]; do
    sleep 20
    # Check for errors
    if grep -qE "segmentation violation|DATA_LIKE_OUTPUT|CANCELLED|please rerun" "$errorloc"; then
        echo "segmentation violation/DATA_LIKE_OUTPUT/CANCELLED error detected!"

        # Kill background processes and restart
        echo "Killing all background processes"
        kill $(jobs -p)

        # Ensure all processes are terminated
        wait

        # Clear error and output files
        echo "Clearing error and output files to prevent resubmission loops"
        > "$errorloc"
        > "$outputloc"

        # Restart AraSim processes
        echo "Restarting AraSim processes"
        for (( i=0; i<${threads}; i++ )); do
            indiv_thread=$((init * threads + i + 1))
			dataoutloc="$TMPDIR/AraOut_${gen}_${indiv}_${indiv_thread}.txt"
            ./AraSim $TMPDIR/setup.txt ${indiv_thread} $TMPDIR > $dataoutloc &
        done
    fi
    
    num_running_processes=$(jobs -r | wc -l) # Number of running processes (AraSim instances in this job)
    echo "Number of threads left: $num_running_processes"
    if [ "$num_running_processes" -eq 0 ]; then
        echo "All $threads threads have completed."
        simulationcheck=True
    fi
done

cd $TMPDIR

echo "Done running AraSim processes"

echo "Moving AraSim outputs to final destination"
for (( i=0; i<${threads}; i++ ))
do
    indiv_thread=$((${init}*${threads}+${i}))
    dataoutloc="$TMPDIR/AraOut_${gen}_${num}_${indiv_thread}.txt"
    #mv AraOut.setup.txt.run${indiv_thread}.root $WorkingDir/Antenna_Performance_Metric/AraOut_${gen}_${num}_${indiv_thread}.root
    rm AraOut.setup.txt.run${indiv_thread}.root
    mv "$dataoutloc" "$WorkingDir/Run_Outputs/$RunName/Generation_Data/${gen}/ara_outputs/AraOut_${gen}_${num}_${indiv_thread}.txt"
done

wait
