# AGENTS.md

Always use the existing conda environment for this project.

Before running python, if you are making codes using machine learning potential, for each mlp, you should use different virtual environment.

for mattersim,
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mattersim_env
```
for mace mp,
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mace_env
```
for sevennet,
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sevenn_env