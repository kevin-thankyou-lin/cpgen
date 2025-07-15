#!/bin/bash


# This script generates images from robomimic based on the specified task and timestep range.
#
# Sample usage:
#   bash launch_generate_images_from_robomimic.sh [task] [start_timestep] [end_timestep]
#   e.g. ./script_name.sh door 91 100

usage() {
    echo "This script generates images from robomimic based on the specified task and timestep range."
    echo "Sample usage:"
    echo "  ./script_name.sh [task] [start_timestep] [end_timestep]"
    echo "  e.g. ./script_name.sh door 91 100"
    exit 1
}

# If number of arguments is not 3, show usage and exit
if [ "$#" -ne 3 ]; then
    echo "Invalid number of arguments: $#, expected 3 arguments" >&2
    usage
fi

# Get the task and timestep range from command line arguments
task="$1"
start_timestep="$2"
end_timestep="$3"

TARGET_OBJECT_NAME="cube"
# Check for task and assign the demo_path accordingly
if [ "$task" = "door" ]; then
    demo_path="../diffusion_policy/data/robomimic/datasets/door/subsampled_image.hdf5"
elif [ "$task" = "lift" ]; then
    demo_path="../diffusion_policy/data/robomimic/datasets/lift/ph/1_demo.hdf5"
elif [ "$task" = "can" ]; then
    TARGET_OBJECT_NAME="can"
    demo_path="../diffusion_policy/data/robomimic/datasets/can/ph/1_demo.hdf5"
elif [ "$task" = "square" ]; then
    demo_path="../diffusion_policy/data/robomimic/datasets/square/ph/1_demo.hdf5"
    TARGET_OBJECT_NAME="square_peg"
    echo "Hardcoding TARGET_OBJECT_NAME=${TARGET_OBJECT_NAME}"
elif [ "$task" = "toolhang" ]; then
    demo_path="../diffusion_policy/data/robomimic/datasets/tool_hang/ph/1_demo.hdf5"
else
    echo "Invalid task: $task"
    usage
    exit 1
fi

# Create a string of timesteps using seq
timesteps=$(seq "$start_timestep" "$end_timestep")

echo "Generating images for task: $task"
echo "Timesteps: $timesteps"


PYTHONPATH=. python scripts/generate_images_from_robomimic.py \
    --cfg.dataset "$demo_path" \
    --cfg.save-dir data/robomimic/"$task"/newest/ \
    --cfg.timesteps ${timesteps[@]} \
    --cfg.num-cameras 20 \
    --cfg.image-height 512 \
    --cfg.image-width 512 \
    --cfg.overwrite-all \
    --cfg.target-obj-name $TARGET_OBJECT_NAME \
    --cfg.dataparser-type nerfstudio-data \
    --cfg.use-target-obj-cameras \
    --cfg.no-use-eye-in-hand-cameras \
    --cfg.no-use-camera-xml-cameras \
    --cfg.radius 0.3 \
