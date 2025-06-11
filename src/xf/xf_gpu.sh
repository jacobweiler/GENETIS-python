#!/bin/bash

## We want to submit the XFsolver as a job array to GPUs
## We'll submit up to 4 at a time (based on number of XF keys)
## Here's the submission command:
## sbatch --array=1-$NPOP%$batch_size --export=ALL,(variables) GPU_XF_job.sh
#SBATCH -A PAS1960
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -n 40
#SBATCH -G 2
#SBATCH --output=Run_Outputs/%x/XF_Outputs/XF_%a.output
#SBATCH --error=Run_Outputs/%x/XF_Errors/XF_%a.error
##SBATCH --mem-per-gpu=178gb

module load xfdtd/7.10.2.3
module load cuda

source $WorkingDir/Run_Outputs/$RunName/setup.sh

individual_number=$((${gen}*${NPOP}+${SLURM_ARRAY_TASK_ID}))

indiv_dir=$XFProj/Simulations/$(printf "%06d" $individual_number)/Run0001

cd $indiv_dir
echo "We are in the directory: $indiv_dir"

pwd
ls

licensecheck=False
simulationcheck=False
while [ $simulationcheck = False ] && [ $licensecheck = False ]; do
	echo "Running XF solver"
	cd $indiv_dir
	xfsolver --use-xstream=true --xstream-use-number=2 --num-threads=2 -v
	
	# Check for unstable calculation in xsolver
	# If unstable, then we need to rerun the simulation
	cd $WorkingDir/Run_Outputs/$RunName/XF_Outputs
	# Adding in check for license error and rerunning until it finds one 
	if [ $(grep -c "Unable to check out license." XF_${SLURM_ARRAY_TASK_ID}.output) -gt 0 ];then
		echo "License error detected. Terminating XFSolver."
		echo "Rerunning XFSolver"
		cp XF_${SLURM_ARRAY_TASK_ID}.output XF_${SLURM_ARRAY_TASK_ID}_${gen}_LICENSE_ERROR.output
		echo " " > XF_${SLURM_ARRAY_TASK_ID}.output
	else
		echo "Solver finished"
		licensecheck=True
	fi
	#check the XF_${SLURM_ARRAY_TASK_ID}.output file for "Unstable calculation detected. Terminating XFSolver."
	# if it's there, then we need to rerun the simulation
	if [ $(grep -c "Unstable calculation detected. Terminating XFSolver." XF_${SLURM_ARRAY_TASK_ID}.output) -gt 0 ];then
		echo "Unstable calculation detected. Terminating XFSolver."
		echo "Rerunning simulation"
		cp XF_${SLURM_ARRAY_TASK_ID}.output ${gen}_XF_${SLURM_ARRAY_TASK_ID}_ERROR.output
		echo " " > XF_${SLURM_ARRAY_TASK_ID}.output
	else
		echo "Simulation finished"
		simulationcheck=True
	fi
done

echo "finished XF solver"

cd $WorkingDir/Run_Outputs/$RunName/GPUFlags

mkdir -m777 $WorkingDir/Run_Outputs/$RunName/uan_files/${gen}_uan_files/$SLURM_ARRAY_TASK_ID 2> /dev/null

echo "The GPU job is done!" >> Part_B_GPU_Flag_${individual_number}.txt 
