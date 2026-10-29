#!/bin/bash
#
#SBATCH --job-name=stanford_hydra-vit-3_layers-3_heads-512_mlp_dim-192_embed_dim
##SBATCH --qos=7d # SCC: --qos=7d, CIDBN: comment it.
#SBATCH -t 24:00:00 # SCC: 7-00, CIDBN: 14-00, 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1 # Single task with multiple CPUs
#SBATCH --cpus-per-task=68 # 68 CPUs per task for multithreaded preprocessing
#SBATCH -A cidbn_wibral_neuro_human ## scc_users // cidbn_legacy
#SBATCH -p cidbn ## medium // cidbn.
#SBATCH --output=stanford_hydra-40_episodes-vit-3_layers-3_heads-512_mlp_dim-192_embed_dim.out #/dev/null
#SBATCH --error=stanford_hydra-40_episodes-vit-3_layers-3_heads-512_mlp_dim-192_embed_dim.err #/dev/null

##SBATCH --mem=480G # 500G

# Load required modules
module load gcc/11.5.0
# module load cuda/11.8.0
module load miniforge3/24.3.0-0

cd /user/fiorenza/u24781/
source activate robotic

echo ---------------------------------------------------------------
echo ---------------------------------------------------------------
which python
echo ---------------------------------------------------------------
echo ---------------------------------------------------------------

# Add project root to Python path so 'model' module is findable
cd /user/fiorenza/u24781/.project/dir.project/users/fiorenza/project_repositories/VLA_FrankaArm/
export PYTHONPATH="/user/fiorenza/u24781/.project/dir.project/users/fiorenza/project_repositories/VLA_FrankaArm:$PYTHONPATH"

######## Run with torchrun for DDP support.
# torchrun spawns processes and handles coordination automatically
# When running under SLURM, torchrun uses SLURM's environment variables
# We use --nproc_per_node=$SLURM_NTASKS to match the single task with cpus-per-task

# Clean up any stale rendezvous files from previous runs
rm -f /tmp/ddp_rendezvous_${SLURM_JOBID}.txt 2>/dev/null

# Use SLURM_NTASKS for --nproc_per_node since we have 1 task with multiple CPUs
# SLURM_NTASKS is set to 1 by --ntasks=1, but we want 68 processes
# So we use --cpus-per-task=68 with --nproc_per_node=68

# Optional: Use preprocessed data if available (skip expensive preprocessing)
# Uncomment and modify the following line to use preprocessed data:
# PREPROCESSED_ARGS="--preprocessed_path preprocessed_data/preprocessed_dataset.pt"

torchrun \
    --nproc_per_node=68 \
    --master_port=29500 \
    -m model.train --settings cfg_stanford_hydra.json --n_epochs 30 --use_ddp $PREPROCESSED_ARGS
EXIT_CODE=$?

# Clean up rendezvous file after training
rm -f /tmp/ddp_rendezvous_${SLURM_JOBID}.txt 2>/dev/null

conda deactivate
exit $EXIT_CODE