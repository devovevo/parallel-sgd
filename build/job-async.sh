#!/bin/bash
#SBATCH -N 8
#SBATCH -C cpu
#SBATCH --qos=debug
#SBATCH -J upctest 
#SBATCH --ntasks-per-node=4
#SBATCH -t 00:10:00


#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export GASNET_BACKTRACE=1

#run the application:
srun ./async_pm

