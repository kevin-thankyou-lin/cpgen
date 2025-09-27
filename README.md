# Constraint-Preserving Data Generation for Visuomotor Policy Generalization

This repo contains the official data generation code for the paper *Constraint-Preserving Data Generation for Visuomotor Policy Generalization*.
See [adaflow-cpgen](https://github.com/kevin-thankyou-lin/adaflow-cpgen) for policy training and evaluation code.

## Installation

Install [mambaforge](https://github.com/conda-forge/miniforge#install) (or conda):

```
mamba create --name cpgen -y python=3.10
mamba activate cpgen

# clone cpgen
git clone https://github.com/kevin-thankyou-lin/cpgen.git  # TODO update name

# manually clone cpgen and mimicgen

git clone https://github.com/NVlabs/mimicgen.git
git clone https://github.com/kevin-thankyou-lin/cpgen-envs.git

pip install -e mimicgen
pip install -e cpgen_envs
pip install -e cpgen

# manually clone curobo
# see https://curobo.org/get_started/1_install_instructions.html
pip install -e curobo --no-build-isolation
```

# Download source datasets
```bash
git lfs install
git clone https://huggingface.co/datasets/cpgen/datasets-src
mv datasets-src/datasets cpgen/src
```

# Download generated datasets
```bash
git lfs install
# NOTE: installing full dataset will take some time. You can install specific dataset using this GIT_LFS_SKIP_SMUDGE=1 flag:
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/cpgen/datasets
# Then, you can pull specific datasets, e.g., : 
git lfs pull -I "datasets/generated/ThreePieceAssemblyWide/"

mv datasets-src/datasets cpgen
# fetch actual data
git lfs pull <path/to/file>
```

# Playback dataset with actions

```bash
# --n 1 means to play back one action only
python scripts/playback_dataset.py --dataset <path/to/hdf5> --use-actions --video_path playback_dataset.mp4  --n 1
```


# Generate data
```
MUJOCO_GL=egl python demo_aug/generate.py --cfg.demo-path datasets/source/square.hdf5 --cfg.env-name Square_D1

MUJOCO_GL=egl python demo_aug/generate.py --cfg.demo-path datasets/source/square.hdf5 --cfg.env-name SquareWide
```

### Data-gen behavior fixing

1. Robot opens gripper after grasping object? Likely because `obj_to_parent_attachment_frame` is not specified.
