#!/bin/bash

CONFIG_DIR="$1"
CONFIG_BASENAME=$(basename "$CONFIG_DIR")

# Generate SLURM script
cat > "launch_script_${CONFIG_BASENAME}.sh" <<EOF
#!/bin/bash
#SBATCH -J $CONFIG_BASENAME
#SBATCH -D ./
#SBATCH -o ./%x.%A_%a.%N.out
#SBATCH -e ./%x.%A_%a.%N.err
#SBATCH --get-user-env
#SBATCH --export=NONE

# Pick cluster + partition (required on LRZ Linux Cluster)
#SBATCH --clusters=serial
#SBATCH --partition=serial_std   # use serial_long for very long runs
#SBATCH --time=14:00:00

# Resources (adjust as needed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

module load slurm_setup
export GRB_LICENSE_FILE=/dss/dsshome1/06/go59sal2/gurobi.lic
module load python/3.10.12-base
source .venv/bin/activate
python script_train_model.py $CONFIG_DIR
deactivate
EOF

chmod +x "launch_script_${CONFIG_BASENAME}.sh"
sbatch "launch_script_${CONFIG_BASENAME}.sh"