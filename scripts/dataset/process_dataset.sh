#/bin/bash

# Processes merged dataset (typically generated using real world data format) to enable policy training

# Define the base input path
RESIZE_INPUT_PATH="/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/wine_glass_hanging/grasp/narrow/demo_025trials/2024-08-12/nullwine_glass_hanging_grasp_narrow_trials25_se3aug-dx-0.05-0.05-dy-0.05-0.05-dz0.0-0.0-dthetz0.0-0.0-biassampzrot_staug-eeposbnd0.12-eerotbnd-40.0,30.0-30.0,30.0-10.0,100.0_defcurobocfg_CUROBO_8f5v7fvj_125demos.hdf5"

# Derive the output paths
RESIZE_OUTPUT_PATH="${RESIZE_INPUT_PATH%.hdf5}-resized.hdf5"
RENAMED_KEYS_OUTPUT_PATH="${RESIZE_OUTPUT_PATH%.hdf5}-renamed.hdf5"

echo "Resizing images"
PYTHONPATH=. python scripts/dataset/resize_images.py --cfg.input-file-path $RESIZE_INPUT_PATH --cfg.output-file-path $RESIZE_OUTPUT_PATH

RENAMED_KEYS_OUTPUT_PATH="${RESIZE_OUTPUT_PATH%.hdf5}-renamed.hdf5"

echo "Renaming keys"
# script that merges training data, resizes them, generates mp4 files and optionally lets use view the h5 structure
PYTHONPATH=. python scripts/dataset/rename_keys.py --cfg.input-file-path $RESIZE_OUTPUT_PATH --cfg.output-file-path $RENAMED_KEYS_OUTPUT_PATH

PYTHONPATH=. python scripts/dataset/view_h5_structure.py --cfg.file-path $RENAMED_KEYS_OUTPUT_PATH

echo "Output path is"
echo $RENAMED_KEYS_OUTPUT_PATH
