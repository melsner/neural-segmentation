#PBS -N autoencode_segment
#PBS -l nodes=1:gpus=1
#PBS -l walltime=48:00:00

module load python/2.7
cd $PBS_O_WORKDIR
source activate local

mkdir -p logs
