#PBS -N autoencode_segment
#PBS -l nodes=1:gpus=2
#PBS -l walltime=120:00:00

module load python/2.7.8
cd $PBS_O_WORKDIR
source activate local

mkdir -p logs
