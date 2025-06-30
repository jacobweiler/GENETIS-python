#!/bin/bash
## This job is designed to be submitted by an array batch submission
## Here's the command:
## sbatch --array=1-NPOP*SEEDS%max --export=ALL,(variables) ara_job.sh
#SBATCH -A PAS1960
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -n 40

# Need to set this up to be used for either HPOL or VPOL

source /fs/ess/PAS1960/BiconeEvolutionOSC/new_root/new_root_setup.sh

cd $AraSimDir

threads=40
num=$(($((${SLURM_ARRAY_TASK_ID}-1))/${Seeds}+1))
seed=$(($((${SLURM_ARRAY_TASK_ID}-1))%${Seeds}+1))
echo "num is $num"
echo "seed is $seed"
# init = seed - 1
init=$((${seed}-1))
echo "init is $init"

echo a_${num}_${seed}.txt

# We need to create a setup file for each individual in the population 
gain_file="${RunDir}/txt_files/a_${num}.txt "

echo "gain file is $gain_file"

sed -e "s|num_nnu|$nnt_per_ara|" -e "s|n_exp|$exp|" -e "s|current_seed|$SpecificSeed|" -e "s|hpol_gain|$gain_file|" ${AraSimExec}/SETUP/setup_dummy_hpol.txt > $TMPDIR/setup.txt

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

wait

cd $TMPDIR

echo "Done running AraSim processes"

echo "Moving AraSim outputs to final destination"
for (( i=0; i<${threads}; i++ ))
do
    indiv_thread=$((${init}*${threads}+${i}))
    dataoutloc="$TMPDIR/AraOut_${gen}_${num}_${indiv_thread}.txt"
    #mv AraOut.setup.txt.run${indiv_thread}.root $WorkingDir/Antenna_Performance_Metric/AraOut_${gen}_${num}_${indiv_thread}.root
    rm AraOut.setup.txt.run${indiv_thread}.root
    mv "$dataoutloc $WorkingDir/Run_Outputs/$RunName/Generation_Data/${gen}/ara_outputs/AraOut_${gen}_${num}_${indiv_thread}.txt"
done

wait
