#!/bin/bash
#
#SBATCH --job-name=stanford_hydra-vit-3_layers-3_heads-512_mlp_dim-192_embed_dim
##SBATCH --qos=7d # SCC: --qos=7d, CIDBN: comment it.
#SBATCH -t 24:00:00 # SCC: 7-00, CIDBN: 14-00, 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12 # SCC: 96, CIDBN: 128
#SBATCH -A cidbn_wibral_neuro_human ## scc_users // cidbn_legacy
#SBATCH -p cidbn ## medium // cidbn.
#SBATCH --output=stanford_hydra-vit-3_layers-3_heads-512_mlp_dim-192_embed_dim.out #/dev/null
#SBATCH --error=stanford_hydra-vit-3_layers-3_heads-512_mlp_dim-192_embed_dim.err #/dev/null

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
# which mpiexec
echo ---------------------------------------------------------------
echo ---------------------------------------------------------------

# Add project root to Python path so 'model' module is findable
cd /user/fiorenza/u24781/.project/dir.project/users/fiorenza/project_repositories/VLA_FrankaArm/
export PYTHONPATH="/user/fiorenza/u24781/.project/dir.project/users/fiorenza/project_repositories/VLA_FrankaArm:$PYTHONPATH"


######## Run multiple datafiles.
python -m model.train --settings cfg_stanford_hydra.json --n_epochs 30
conda deactivate