#!/bin/bash

# Note: run this in a tmux session
# Also, to resume training or load from checkpoint, ensure the date is correct and update the HYDRA_RUN_DIR
# if eval'ing only, use a new wandb logging id to avoid updating logs from the resumed run



# if bohg-ws-16, base_dir is /src/thankyou
if [[ $(hostname) == *bohg-ws-16* || $(hostname) == *bohg-ws-17* ]]; then
    BASE_DIR="/scr/thankyou"
    ENV_SRC_CMD="source ${BASE_DIR}/.bashrc"
    source ${BASE_DIR}/.bashrc
else
    BASE_DIR="/juno/u/thankyou"
    ENV_SRC_CMD="source ${BASE_DIR}/.bashrc.user"
    source ${BASE_DIR}/.bashrc.user
    echo "in juno host"
fi

current_hostname=$(hostname)

VIRTUAL_ENV_CMD=""
if [[ $current_hostname != *bohg-ws-13* && $current_hostname != *bohg-ws-14* && $current_hostname != *bohg-ws-16* && $current_hostname != *bohg-ws-17* && $current_hostname != *juno2* ]]; then
    conda activate acta-wo-tcnn
    echo "Activated acta-wo-tcnn"
    VIRTUAL_ENV_CMD="${ENV_SRC_CMD} && conda activate acta"
else
    conda activate acta
    echo "Activated acta"
    VIRTUAL_ENV_CMD="${ENV_SRC_CMD} && conda activate acta"
fi


# Define a variable to control whether each block should be executed
generate=false
merge=true
train=true
debug=false

TRIALS_PER_SEED=25
CMDs_PER_GPU=2
SEEDS_PER_CMD=8

if [ "$debug" = true ]; then
    TRIALS_PER_SEED=1
    CMDs_PER_GPU=1
    SEEDS_PER_CMD=1
fi

TASK_NAME="lift"
SUBTASK_NAME="grasp"
TASK_DISTRIBUTION="wide"

case "$TASK_DISTRIBUTION" in
    "narrow"|"wide"|"narrow-debug"|"narrow-debug-w-scale"|"narrow-debug-w-scale-opp-rot"|"full-distribution")
        ;;
    *)
        echo "Invalid task distribution: $TASK_DISTRIBUTION. Must be narrow or wide or narrow-debug."
        exit 1
        ;;
esac

TIME_PER_AUG_TRIAL=20  # multiple nerfs makes things slow

current_date=$(date +"%Y-%m-%d")
echo "Current date: ${current_date}"

aug_data_dir="/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/${TASK_NAME}/${SUBTASK_NAME}/${TASK_DISTRIBUTION}/demo_0${TRIALS_PER_SEED}trials/${current_date}"

BASE_SEED=0
##########################
# Generate augmentations #
##########################

N_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Number of gpus: ${N_GPUS}"
TOTAL_AUGS=$((N_GPUS * CMDs_PER_GPU  * SEEDS_PER_CMD * TRIALS_PER_SEED))
echo "Total number of augmentations: ${TOTAL_AUGS}"

echo $TRIALS_PER_SEED
echo $SEEDS_PER_CMD
echo $CMDs_PER_GPU
echo $N_GPUS

if [ "$generate" = true ]; then
    SESSION_PREFIX="acta_"

    echo "Launching generate_aug.sh with $N_GPUS gpus"
    echo "Augmentation dataset directory: ${aug_data_dir}"

    for (( gpu=0; gpu<$N_GPUS; gpu++ )); do
        for (( cmd_idx=0; cmd_idx<$CMDs_PER_GPU; cmd_idx++ )); do
            # Calculate the seed start and end values.
            START_SEED=$((BASE_SEED + SEEDS_PER_CMD*cmd_idx + SEEDS_PER_CMD*CMDs_PER_GPU*gpu))
            END_SEED=$((START_SEED + SEEDS_PER_CMD - 1))

            # Construct the session name.
            SESSION_NAME=${SESSION_PREFIX}$((CMDs_PER_GPU*gpu + cmd_idx + BASE_SEED))

            # Construct the command.
            CMD="CUDA_VISIBLE_DEVICES=$gpu bash scripts/launch_generate_aug.sh $START_SEED $END_SEED ${TASK_NAME} ${SUBTASK_NAME} ${TASK_DISTRIBUTION} $TRIALS_PER_SEED"

            # Echo the command for verification.
            echo $CMD
            echo $SESSION_NAME

            # Create the tmux session without executing anything.
            tmux new-session -d -s $SESSION_NAME
            sleep 0.5

            # send virtual env creation command
            tmux send-keys -t $SESSION_NAME "$VIRTUAL_ENV_CMD" C-m
            sleep 0.5

            # send directory change command
            tmux send-keys -t $SESSION_NAME "cd ${BASE_DIR}/autom/demo-aug/" C-m
            sleep 0.5

            # Send the command to the tmux session.
            tmux send-keys -t $SESSION_NAME "$CMD" C-m
            sleep 5
        done
    done

    BUFFER_SLEEP_TIME=100
    total_sleep_time=$((SEEDS_PER_CMD * TRIALS_PER_SEED * TIME_PER_AUG_TRIAL  + BUFFER_SLEEP_TIME))
    echo "Sleeping for ${total_sleep_time} secs [$(($total_sleep_time / 60)) min(s)] to allow all tasks to complete."

    sleep $total_sleep_time
    echo "All tasks completed. Continuing with the script."
fi

###############################################
# Merge the generated augs into a single file #
###############################################

if [ "$merge" = true ]; then
    PYTHONPATH=. python scripts/merge_demo_hdf5s.py --cfg.src-dir ${aug_data_dir}
    echo "Merging completed. Continuing with training."
fi


################
# Train policy #
################

HYDRA_RUN_DIR="data/outputs/$(date +"%Y-%m-%d")/$(date +"%H-%M-%S")_${TASK_NAME}"

echo "Hydra run dir: ${HYDRA_RUN_DIR}"

if [ "$train" = true ]; then
    # Assumption: merge_demo_hdf5.py saves merged file to ${aug_data_dir}
    merged_data_path="${aug_data_dir}/*.hdf5"
    echo "Merged data path: ${merged_data_path}"

    # Get a list of matching files
    hdf5_files=$(ls $merged_data_path)

    # Assert that there is only one .hdf5 file
    if [ $(echo "$hdf5_files" | wc -l) -ne 1 ]; then
        echo "Error: there must be exactly one .hdf5 file in $aug_data_dir"
        exit 1
    fi

    merged_data_path="${hdf5_files[0]}"
    echo "Merged data path: ${merged_data_path}"

    mamba activate robodiff-dp

    cd ${BASE_DIR}/autom/consistency-policy

    echo "HYDRA_FULL_ERROR=1 python train.py --config-dir=configs/ \
    --config-name=dp_unet_${TASK_NAME}.yaml \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    task.dataset.dataset_path=$merged_data_path \
    task.dataset_path=$merged_data_path \
    task.env_runner.dataset_path=$merged_data_path \
    logging.name=$(basename $merged_data_path .$(echo $merged_data_path | rev | cut -d. -f1 | rev))"

    base_name=$(basename $merged_data_path .$(echo $merged_data_path | rev | cut -d. -f1 | rev))

    # lift task
    if [ $TASK_NAME == "lift" ]; then
        if [ $TASK_DISTRIBUTION == "narrow" ]; then
            randomize_block_pos=false
            randomize_shapes=false
        else
            randomize_block_pos=true
            randomize_shapes=true
            block_x_range='[-0.1,0.1]'
            block_y_range='[-0.1,0.1]'
            block_min='[0.014,0.014,0.014]'  # x0.8
            block_max='[0.024,0.024,0.024]'  # x1.2; this range gives 0.8 success rate
        fi
        randomization_params_name="${block_min}_block_max_${block_max}_block_x_range_${block_x_range}_block_y_range_${block_y_range}"
    elif [ $TASK_NAME == "square" ]; then
        x_range='[-0.115,-0.11]'
        y_range='[0.11,0.225]'
        rot_range='null'
        scale_range='null'
        use_scale_variation=false
        if [ $TASK_DISTRIBUTION == "narrow" ]; then
            use_scale_variation=false
        elif [ $TASK_DISTRIBUTION == "narrow-debug" ]; then
            use_scale_variation=false
            rot_range='[1.57,4.712]'
        elif [ $TASK_DISTRIBUTION == "wide" ]; then
            use_scale_variation=true
            scale_range='[0.8,1.1]'  # x0.8
        elif [ $TASK_DISTRIBUTION == "narrow-debug-w-scale" ]; then
            use_scale_variation=true
            scale_range='[0.8,1.1]'  # x0.8
            rot_range='[1.57,4.712]'
        elif [ $TASK_DISTRIBUTION == "full-distribution" ]; then
            use_scale_variation=true
            scale_range='[0.8,1.1]'  # x0.8
            rot_range='null'
        else
            echo "Invalid task distribution: $TASK_DISTRIBUTION. Must be narrow or wide."
            exit 1
        fi
        randomization_params_name="obj_xrng-${x_range}_yrng-${y_range}_rotrng-${rot_range}"
        if [ $use_scale_variation == true ]; then
            randomization_params_name="${randomization_params_name}_scalerng-${scale_range}"
        fi
    elif [ $TASK_NAME == "can" ]; then
        randomize_block_pos=false
        randomize_shapes=false
    else
        echo "Invalid task: $TASK_NAME. Must be door, lift, can, or square."
        exit 1
    fi

    # "task.env_runner.randomize_block_pos=${randomize_block_pos}" \
    # "task.env_runner.block_x_range=${block_x_range}" \
    # "task.env_runner.block_y_range=${block_y_range}" \
    # "task.env_runner.randomize_shapes=${randomize_shapes}" \
    # "task.env_runner.block_min=${block_min}" \
    # "task.env_runner.block_max=${block_max}" \

    # "task.env_runner.x_range=${x_range}" \
    # "task.env_runner.y_range=${y_range}" \
    # "task.env_runner.rot_range=${rot_range}" \
    # "task.env_runner.use_scale_variation=${use_scale_variation}" \
    # "task.env_runner.scale_range=${scale_range}" \


    sanitized_randomization_params_name=$(echo "$randomization_params_name" | tr ',' '-' | tr -d '[]')
    random_id=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | head -c 8)

    overall_name="${TASK_NAME}_${SUBTASK_NAME}_${TASK_DISTRIBUTION}_${base_name}_${sanitized_randomization_params_name}"
    echo $overall_name
    echo $randomization_params_name

    HYDRA_FULL_ERROR=1  PYTHONPATH=.:../demo-aug/ python train.py --config-dir=../consistency-policy/configs/old_configs/ \
    --config-name=dp_unet_${TASK_NAME}.yaml \
    hydra.run.dir=${HYDRA_RUN_DIR} \
    task.dataset.dataset_path=$merged_data_path \
    task.dataset_path=$merged_data_path \
    task.env_runner.dataset_path=$merged_data_path \
    training.rollout_every=50 \
    training.num_epochs=601 \
    training.eval_policy_once_only=false \
    training.resume=true \
    training.debug=false \
    task.env_runner.n_envs=10 \
    task.env_runner.n_test=50 \
    task.env_runner.n_train=0 \
    task.env_runner.n_test_vis=50 \
    task.env_runner.n_train_vis=0 \
    task.env_runner.max_steps=90 \
    logging.name=$overall_name \
    logging.id=$random_id  # add random id to log to different wandb run than the resumed
fi

# Note: need to fix bug where currently all envs have env_type hardcoded. However, this means a 'test' env init might be different from 'train' env init
# done
echo "Done"
