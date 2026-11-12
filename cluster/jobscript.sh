#!/bin/bash
#
#SBATCH --job-name=stanford_hydra-vit-3_layers-3_heads-512_mlp_dim-192_embed_dim
##SBATCH --qos=7d # SCC: --qos=7d, CIDBN: comment it.
#SBATCH -t 24:00:00 # SCC: 7-00, CIDBN: 14-00, 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128 # scc-cpu: max_cpu=96 ; cidbn: max_cpu=128
#SBATCH -A cidbn_wibral_neuro_human
#SBATCH -p cidbn
#SBATCH --output=stanford_hydra-%j.out
#SBATCH --error=stanford_hydra-%j.err

# Set environment variables
NPROC_PER_NODE=32 # Defines the number of DDP processes spanwed.
CPUS_PER_RANK=$((SLURM_CPUS_PER_TASK / NPROC_PER_NODE)) # CPU threads per DDP process
export OMP_NUM_THREADS=$CPUS_PER_RANK
export MKL_NUM_THREADS=$CPUS_PER_RANK
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 # Python normally buffers standard output when it is redirected to a file, such as your Slurm .out file. Prints may exist in the process's output buffer without appearing immediately in the file.
MASTER_PORT=$((15000 + SLURM_JOB_ID % 40000))


echo "JOB START: $(date)"
echo "JOB ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "DDP processes: $NPROC_PER_NODE"
echo "OMP threads: $OMP_NUM_THREADS"

# Load required modules
module load gcc/11.5.0
module load miniforge3/24.3.0-0
# module load cuda/11.8.0

echo "Modules loaded: $(date)"

cd /user/fiorenza/u24781/
source activate robotic

echo "Environment activated: $(date)"
which python
python --version

# Add project root to Python path
cd /user/fiorenza/u24781/.project/dir.project/users/fiorenza/project_repositories/VLA_FrankaArm/
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "Launching torchrun: $(date)"

# Clean up any stale rendezvous files
rm -f /tmp/ddp_rendezvous_${SLURM_JOBID}.txt 2>/dev/null

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    -m model.train --settings cfg_stanford_hydra.json --n_epochs 30 --use_ddp $PREPROCESSED_ARGS 2>&1
EXIT_CODE=$?

# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=16 \
#     test/ddp_test.py



rm -f /tmp/ddp_rendezvous_${SLURM_JOBID}.txt 2>/dev/null
conda deactivate
exit $EXIT_CODE