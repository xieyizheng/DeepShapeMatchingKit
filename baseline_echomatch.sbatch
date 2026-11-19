#!/bin/bash
#SBATCH --job-name=echo-baseline
#SBATCH --partition=mlgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

# Load the Python environment on Marvin
module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"

# Run from the directory where sbatch was submitted
cd "$SLURM_SUBMIT_DIR"

# Evaluation environment
export WANDB_MODE=disabled
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4

echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Commit: $(git rev-parse --short HEAD)"

nvidia-smi

config=options/echo_match/test/echo_match_psmal_xyz.yaml
python - "$config" <<'PY'
import sys
from pathlib import Path

import yaml

options = yaml.safe_load(Path(sys.argv[1]).read_text())
checkpoint = Path(options["path"]["resume_state"]).expanduser()
if not checkpoint.is_file():
    sys.exit(f"Baseline checkpoint is missing: {checkpoint}")
print(f"Baseline checkpoint: {checkpoint.resolve()}")
PY

python -u test.py --opt "$config"

echo "Baseline completed successfully"
