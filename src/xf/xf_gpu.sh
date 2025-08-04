#!/bin/bash
#SBATCH -A PAS1960
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH -n 40
#SBATCH -G 2
#SBATCH --output=Run_Outputs/%x/job_outs/xf_out/XF_%a.output
#SBATCH --error=Run_Outputs/%x/job_outs/xf_err/XF_%a.error
##SBATCH --mem-per-gpu=178gb

module load xfdtd/7.11.0.3
module load cuda/12.6.2

individual_number=$((${gen}*${NPOP}+${SLURM_ARRAY_TASK_ID}))
indiv_dir=$XFProj/Simulations/$(printf "%06d" $individual_number)/Run0001

cd $indiv_dir
echo "We are in the directory: $indiv_dir"

licensecheck=False
simulationcheck=False
while [ $simulationcheck = False ] && [ $licensecheck = False ]; do
	echo "Running XF solver"
	cd $indiv_dir
	xfsolver --use-xstream=true --xstream-use-number=2 --num-threads=2 -v
	
	# Check for unstable calculation in xsolver
	# If unstable, then we need to rerun the simulation
	cd $WorkingDir/Run_Outputs/$RunName/job_outs/xf_out
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
