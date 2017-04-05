#PBS -N autoencode_segment
#PBS -l nodes=1:gpus=1
#PBS -l mem=12gb
#PBS -l walltime=48:00:00

module load python/2.7
module load cuda
cd $PBS_O_WORKDIR
source activate local
export KERAS_BACKEND=tensorflow

mkdir -p logs
