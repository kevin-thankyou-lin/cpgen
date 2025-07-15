#!/bin/bash


usage() {
    echo "Usage: $0 [gen|merge|train|<combination>] <experiment_name> <env_name> <gpu_id>"
    echo "  Options:"
    echo "      gen                Run data generation"
    echo "      merge              Merge generated data"
    echo "      train              Run training"
    echo "      gen-merge-train Perform data generation, merge the data, and train (order doesn't matter)"
    echo
    echo "  You can specify multiple actions in any order by chaining them with dashes (-). For example:"
    echo "  bash $0 gen-merge-train"
    echo
    echo "  <experiment_name>: Specify the experiment name to associate with the actions."
    echo
    echo "  All specified actions will be triggered, regardless of their order."
    echo
}

usage

if [[ $(hostname) == *cybertron2* ]]; then
    BASE_DIR="/home/kevin"
    source ${BASE_DIR}/.bashrc
elif [[ $(hostname) == *cybertron3* ]]; then
    BASE_DIR="/mnt/data1/kevin/"
    source ${BASE_DIR}/.bashrc
elif [[ $(hostname) == *thankyou* ]]; then
    BASE_DIR="/home/thankyou"
    source ${BASE_DIR}/.bashrc
fi


# N_PROCESSES=4  # N_PROCESSES * N_DEMOS_PER_PROCESS = TOTAL_DEMOS
# N_DEMOS_PER_PROCESS=1
# N_PROCESSES_PER_ROUND=4  # used to spread out runs if needed e.g. due to RAM issues

N_PROCESSES=1  # N_PROCESSES * N_DEMOS_PER_PROCESS = TOTAL_DEMOS
N_DEMOS_PER_PROCESS=1
N_PROCESSES_PER_ROUND=1  # used to spread out runs if needed e.g. due to RAM issues

SPLIT_SRC_DEMOS=false  # Set to true to use process-based splitting of src demo loading; false to use defaults
GPU_LIST=($4)  # List of available GPUs
N_GPUS=${#GPU_LIST[@]}
gpu_id=$4

exp_name=$2
env_name=$3
if [[ -z "$exp_name" ]]; then
    echo "Experiment name is not provided. Exiting..."
    exit 1
fi
ROBOT_TYPE="franka"
# Check if env_name contains "Real" and ROBOT_TYPE is "franka"
if [[ "$env_name" == *Real* ]] && [[ "$ROBOT_TYPE" == "franka" ]]; then
    read -p "You specified a real environment with robot type 'franka'. Did you mean to use 'franka_umi' instead? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ROBOT_TYPE="franka_umi"
        echo "Robot type changed to 'franka_umi'"
    fi
fi

current_datetime=$(date '+%Y-%m-%d-%H-%M-%S')
datagen_save_dir_base="${PWD}"
policy_train_save_dir=null
policy_train_save_dir_base="${PWD}"
if [[ $(hostname) == *cybertron2* ]]; then
    echo "Running on Cybertron2, updating save directories"
    datagen_save_dir_base="/mnt/data3/kevin/autom/demo-aug/"
    policy_train_save_dir_base="/mnt/data3/kevin/autom/adaflow/"
elif [[ $(hostname) == *cybertron3* ]]; then
    echo "Running on Cybertron2, updating save directories"
    datagen_save_dir_base="/mnt/data1/kevin/autom/demo-aug/"
    policy_train_save_dir_base="/mnt/data1/kevin/autom/adaflow/"
fi
base_save_dir="${datagen_save_dir_base}/datasets/generated/${env_name}/${current_datetime}"
final_merged_path="${base_save_dir}/merged_${exp_name}_all.hdf5"
final_merged_no_obs_path="${base_save_dir}/merged_${exp_name}_regular.hdf5"

# Create arrays to store paths of generated datasets
declare -a paths_to_merge=()
declare -a paths_with_obs_to_merge=()

tmp_file_merge_paths=$(mktemp)
tmp_file_merge_paths_w_obs=$(mktemp)

# Check for invalid combination
if [[ "$1" == *"render-depth-seg"* && "$1" != *"render-rgb"* && "$1" == *"train"* ]]; then
    echo "WARNING: You are trying to use 'render-depth-seg' and 'train' without 'render-rgb'."
    echo "This combination won't work correctly. Please include 'render-rgb' in your command or update the run script."
    echo "Example: $0 gen-merge-add-noise-render-rgb-render-depth-seg-train $exp_name $env_name $gpu_id"
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi


# manual paths
# echo "..." >> "$tmp_file_merge_paths"
# echo "..." >> "$tmp_file_merge_paths_w_obs"

# TODO: a) update envs to all load from 0 and 1.
#       b) add all real envs
load_demos_start_idx=0
load_demos_end_idx=1
case "$env_name" in
    *StackThreeReal*)
        demo_path="datasets/source/stack_three_real.hdf5"
        ;;
    *NutAssemblySquareReal*)
        demo_path="datasets/source/nut_assembly_square_real.hdf5"
        ;;
    *ThreePieceAssemblyReal*)
        demo_path="datasets/source/three_piece_assembly_real.hdf5"
        ;;
    *MugCleanupReal*)
        demo_path="datasets/source/mug_cleanup_real.hdf5"
        ;;
    *HammerCleanupReal*)
        demo_path="datasets/source/hammer_cleanup_real.hdf5"
        ;;
    # Non-real environments
    *StackThree*)
        demo_path="datasets/source/stack_three.hdf5"
        ;;
    *Square*)
        demo_path="datasets/source/square.hdf5"
        ;;
    *ThreePieceAssembly*)
        demo_path="datasets/source/three_piece_assembly.hdf5"
        ;;
    *MugCleanup*)
        demo_path="datasets/source/mug_cleanup.hdf5"
        ;;
    *HammerCleanup*)
        demo_path="datasets/source/hammer_cleanup.hdf5"
        ;;
    *Kitchen*)
        demo_path="datasets/source/kitchen.hdf5"
        ;;
    *Coffee*)
        demo_path="datasets/source/coffee.hdf5"
        ;;
    *Threading*)
        demo_path="datasets/source/threading.hdf5"
        ;;
    *)
        echo "Unsupported env_name: $env_name"
        exit 1
        ;;
esac

# Function to run data generation for a single process
run_data_generation() {
    local process_id=$1
    local gpu_id=${GPU_LIST[$((process_id % N_GPUS))]}  # Assign GPUs from list in round-robin fashion
    echo "Using gpu=$gpu_id for data generation"
    local process_save_dir="${base_save_dir}/process${process_id}"
    local merge_demo_save_path="${process_save_dir}/${exp_name}_${process_id}.hdf5"
    local merge_demo_save_path_with_obs="${process_save_dir}/${exp_name}_${process_id}_obs.hdf5"

    # Create process-specific directories
    mkdir -p "${process_save_dir}/failures"
    mkdir -p "${process_save_dir}/motion_plans"
    mkdir -p "${process_save_dir}/successes"
    mkdir -p "${process_save_dir}/videos"

    echo "Running process ${process_id} with save directory: ${process_save_dir}"

    # Use process-specific start/end indices if SPLIT_SRC_DEMOS is true
    if $SPLIT_SRC_DEMOS; then
        local load_demos_start_idx=$((process_id * N_DEMOS_PER_PROCESS))
        local load_demos_end_idx=$((load_demos_start_idx + N_DEMOS_PER_PROCESS))
        echo "Process ${process_id} handling demos from $load_demos_start_idx to $load_demos_end_idx"
    fi

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu_id mamba run -n cpgen python demo_aug/generate.py \
        --cfg.demo-path $demo_path \
        --cfg.load-demos-start-idx $load_demos_start_idx \
        --cfg.load-demos-end-idx $load_demos_end_idx \
        --cfg.controller-type ik \
        --cfg.n-demos ${N_DEMOS_PER_PROCESS} \
        --cfg.require-n-demos \
        --cfg.save-dir $process_save_dir \
        --cfg.env-name $env_name \
        --cfg.merge-demo-save-path $merge_demo_save_path \
        --cfg.seed $((process_id + 1)) \
        --cfg.wandb-name-prefix "${exp_name}_process_${process_id}" \
        --cfg.no-use-reset-near-constraint \
        --cfg.no-gen-single-stage-only \
        --cfg.no-store-single-stage-only \
        --cfg.store-single-stage-max-extra-steps 16 \
        --cfg.constraint-selection-method random \
        --cfg.curobo-goal-type pose_wxyz_xyz \
        --cfg.use-wandb \
        --cfg.wandb-project cpgen \
        --cfg.no-use-reset-to-state \
        --cfg.initialization.initialization-noise-magnitude 0.06 \
        --cfg.initialization.initialization-noise-type gaussian \
        --cfg.robot-type ${ROBOT_TYPE} \
        --cfg.demo-segmentation-type llm-success

    # Write paths to the temporary files
    echo "$merge_demo_save_path" >> "$tmp_file_merge_paths"
    echo "$merge_demo_save_path_with_obs" >> "$tmp_file_merge_paths_w_obs"
}

# Function to merge datasets
merge_datasets() {
    echo "Merging datasets from all processes"

    # Aggregate paths from the temporary files
    mapfile -t paths_to_merge < "$tmp_file_merge_paths"
    mapfile -t paths_with_obs_to_merge < "$tmp_file_merge_paths_w_obs"

    # Print all paths that will be merged
    echo "Regular paths to merge:"
    for path in "${paths_to_merge[@]}"; do
        echo "  - $path"
    done

    echo "Paths with observations to merge:"
    for path in "${paths_with_obs_to_merge[@]}"; do
        echo "  - $path"
    done


    # Convert arrays to space-separated strings for the merge script
    paths_string="${paths_to_merge[*]}"
    paths_with_obs_string="${paths_with_obs_to_merge[*]}"

    # Merge regular paths
    mamba run -n cpgen python scripts/merge_demo_hdf5s.py \
        --cfg.src-paths $paths_string \
        --cfg.save-path "${base_save_dir}/merged_${exp_name}_regular.hdf5"

    # Merge paths with observations
    mamba run -n cpgen python scripts/merge_demo_hdf5s.py \
        --cfg.src-paths $paths_with_obs_string \
        --cfg.save-path $final_merged_path

    echo "Datasets merged successfully to: $final_merged_path"

    # CUDA_VISIBLE_DEVICES=$gpu_id python scripts/dataset/mp4_from_h5.py --cfg.h5-file-path $final_merged_path --cfg.all-demos
}

# === Run if 'gen' is passed as an argument ===
if [[ "$1" == *"gen"* ]]; then
    echo "Running data generation with ${N_PROCESSES} processes on GPU $gpu_id"
    echo "Each process generates ${N_DEMOS_PER_PROCESS} demos"
    echo "Total demos to generate: $((N_PROCESSES * N_DEMOS_PER_PROCESS))"

    mkdir -p "$base_save_dir"

    rounds=$(( (N_PROCESSES + N_PROCESSES_PER_ROUND - 1) / N_PROCESSES_PER_ROUND ))
    echo "Total rounds: $rounds"

    for ((round=0; round<rounds; round++)); do
        echo "=== Starting round $((round + 1)) of $rounds ==="
        for ((j=0; j<N_PROCESSES_PER_ROUND; j++)); do
            i=$((round * N_PROCESSES_PER_ROUND + j))
            if (( i >= N_PROCESSES )); then
                break
            fi
            echo "Launching process $i"
            run_data_generation "$i" &
            sleep 2
        done
        wait
        echo "--- Round $((round + 1)) completed ---"
    done

    echo "All data generation processes completed"

    if [ "$N_PROCESSES" -gt 1 ]; then
        echo "Multiple processes detected, automatically merging datasets..."
        merge_datasets
    fi
fi

if [[ "$1" == *"merge"* ]]; then
    # Only execute explicit merge if it was requested
    merge_datasets
fi

if [[ "$1" == *"add-noise"* ]]; then
    echo "Adding noise to dataset"  # note this variant is meant for latter project ...
    action_noise_std=0.03
    num_processes=12
    num_retries=6
    CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONPATH=. mamba run -n cpgen python scripts/playback_dataset_w_noise_mp.py \
    --dataset $final_merged_no_obs_path \
    --num_retries $num_retries --use-actions --num_processes $num_processes --action_noise_std $action_noise_std \
    --save_demo_dir ${base_save_dir}/${exp_name}/

    final_merged_path="${base_save_dir}/${exp_name}/merged_success_demos_original_actions_noisy_state.hdf5"
fi

if [[ "$1" == *"render-rgb"* ]]; then
    cd ../adaflow

    # could be good to have noising in the run script too ...
    echo "Converting dataset states to observations with rgb"
    CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONPATH=. mamba run -n adaflow python scripts/dataset_states_to_obs.py \
    --dataset $final_merged_path  \
    --output_name ${exp_name}-original-action-noisy-state-action-std-${action_noise_std}-rgb-84-84.hdf5 --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
    --compress --exclude-next-obs

    final_merged_path="${base_save_dir}/${exp_name}/${exp_name}-original-action-noisy-state-action-std-${action_noise_std}-rgb-84-84.hdf5"
    echo "Using dataset path for training: $final_merged_path"
fi

if [[ "$1" == *"render-depth-seg"* ]]; then
    action_noise_std=0.03
    num_processes=12
    num_retries=6
    src_data_path_to_render="${base_save_dir}/${exp_name}/merged_success_demos_original_actions_noisy_state.hdf5"
    cd ../adaflow

    echo "Using dataset path for training: $src_data_path_to_render"
    echo "action noise std: $action_noise_std"

    # check if has train-real thenset height and width to 90 and 160
    if [[ "$1" == *"train-real"* ]]; then
        camera_height=90
        camera_width=160
    else
        camera_height=84
        camera_width=84
    fi

    echo "output_name: ${exp_name}-original-action-noisy-state-action-std-${action_noise_std}-depth-seg-${camera_height}-${camera_height}.hdf5"
    # could be good to have noising in the run script too ...
    echo "Converting dataset states to observations with depth and segmentation"
    CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONPATH=. mamba run -n adaflow python scripts/dataset_states_to_obs.py \
    --dataset $src_data_path_to_render  \
    --output_name ${exp_name}-original-action-noisy-state-action-std-${action_noise_std}-depth-seg-${camera_height}-${camera_width}.hdf5 --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand --camera_height ${camera_height} --camera_width ${camera_width} \
    --depth --segmentation instance --compress --exclude-next-obs

    final_merged_path_depth_seg="${base_save_dir}/${exp_name}/${exp_name}-original-action-noisy-state-action-std-${action_noise_std}-depth-seg-${camera_height}-${camera_width}.hdf5"
    echo "Using dataset path for training: $final_merged_path_depth_seg"
fi

if [[ "$1" == *"train"* ]]; then
    echo "Running training pipeline"

    # Determine which dataset path to use for training
    if [ "$N_PROCESSES" -gt 1 ]; then
        training_dataset_path=$final_merged_path
    else
        # For single process, use the only generated dataset
        training_dataset_path="${paths_with_obs_to_merge[0]}"
    fi

    echo "Using dataset path for training: $training_dataset_path"

    cd ../adaflow

    policy_train_save_dir="${policy_train_save_dir_base}/policy_checkpoints/${env_name}/${current_datetime}_${exp_name}"
    wandb_name="${exp_name}_${env_name}_${current_datetime}"

    max_steps=450
    # update to match file names
    if [[ "$env_name" == *ThreePieceAssembly* ]]; then
        env_name="three_piece_assembly"
        max_steps=600
    elif [[ "$env_name" == *StackThree* ]]; then
        env_name="stack_three"
        max_steps=600
    elif [[ "$env_name" == *Square* ]]; then
        env_name="square"
    elif [[ "$env_name" == *MugCleanup* ]]; then
        env_name="mug_cleanup"
    elif [[ "$env_name" == *HammerCleanup* ]]; then
        env_name="hammer_cleanup"
        max_steps=600
    elif [[ "$env_name" == *PickPlace* ]]; then
        env_name="pick_place"
        max_steps=900
    elif [[ "$env_name" == *Coffee_* ]]; then
        env_name="coffee"
    elif [[ "$env_name" == *CoffeePreparation* ]]; then
        env_name="coffee"
        max_steps=900
    elif [[ "$env_name" == *Threading* ]]; then
        env_name="threading"
    elif [[ "$env_name" == *Kitchen* ]]; then
        env_name="kitchen"
        max_steps=1000
    fi

    # check if has train-real
    if [[ "$1" == *"train-real"* ]]; then
        echo "Also training on real data"
        CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=.:../robosuite HYDRA_FULL_ERROR=1 mamba run -n adaflow python train.py \
            --config-name=train_diffusion_unet_ddpm_image_workspace_robomimic_depth_seg_square_real \
            task=default_depth_seg_real \
            task.dataset_type=ph \
            task.dataset_path=$final_merged_path_depth_seg \
            task.env_runner.n_train=1 \
            task.env_runner.n_train_vis=1 \
            task.env_runner.n_test=14 \
            task.env_runner.n_test_vis=14 \
            task.env_runner.n_envs=15 \
            task.env_runner.dataset_path=$final_merged_path_depth_seg \
            task.env_runner.max_steps=$max_steps \
            task.dataset.dataset_path=$final_merged_path_depth_seg \
            multi_run.run_dir=$policy_train_save_dir \
            logging.name=${wandb_name} \
            hydra.run.dir=$policy_train_save_dir \
            hydra.sweep.dir=$policy_train_save_dir \
            training.num_epochs=250
    else
        echo "Not training for sim2real"
        # use square_image_abs because they're all the same after adjusting max_steps
        CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=.:../robosuite HYDRA_FULL_ERROR=1 mamba run -n adaflow python train.py \
            --config-name=train_diffusion_unet_ddpm_image_workspace_robomimic \
            task=square_image_abs \
            task.dataset_type=ph \
            task.dataset_path=$training_dataset_path \
            task.env_runner.n_train=2 \
            task.env_runner.n_train_vis=2 \
            task.env_runner.n_envs=26 \
            task.env_runner.dataset_path=$training_dataset_path \
            task.env_runner.max_steps=$max_steps \
            task.dataset.dataset_path=$training_dataset_path \
            multi_run.run_dir=$policy_train_save_dir \
            logging.name=${wandb_name} \
            hydra.run.dir=$policy_train_save_dir \
            hydra.sweep.dir=$policy_train_save_dir \
            training.num_epochs=250

        # TODO: not quite correct yet; non-train-real and depth seg not correctly supported
        if [[ "$1" == *"render-depth-seg"* ]]; then
            current_datetime=$(date '+%Y-%m-%d-%H-%M-%S')
            exp_name="${exp_name}_depth_seg"
            policy_train_save_dir="${policy_train_save_dir_base}/policy_checkpoints/${env_name}/${current_datetime}_${exp_name}"
            wandb_name="${exp_name}_${env_name}_${current_datetime}"

            agentview_depth_clamp_max=1.75
            agentview_depth_mean=0.8
            agentview_depth_std=0.35
            # Set depth encoder parameters based on environment type
            if [[ "$env_name" == "stack_three" || "$env_name" == "three_piece_assembly" || "$env_name" == "mug_cleanup" || "$env_name" == "threading" ]]; then
                echo "Setting depth encoder parameters for $env_name environment"
                agentview_depth_mean=1.15
                agentview_depth_std=0.38
            elif [[ "$env_name" == "hammer_cleanup" || "$env_name" == "kitchen" ]]; then
                echo "Setting depth encoder parameters for $env_name environment"
                agentview_depth_mean=0.96
                agentview_depth_std=0.38
            fi
            echo "Also training on depth and segmentation"
            CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=.:../robosuite HYDRA_FULL_ERROR=1 mamba run -n adaflow python train.py \
                --config-name=train_diffusion_unet_ddpm_image_workspace_robomimic_depth_seg \
                task=default_depth_seg_abs \
                task.dataset_type=ph \
                task.dataset_path=$final_merged_path_depth_seg \
                task.env_runner.n_train=2 \
                task.env_runner.n_train_vis=2 \
                task.env_runner.n_envs=26 \
                task.env_runner.dataset_path=$final_merged_path_depth_seg \
                task.env_runner.max_steps=$max_steps \
                task.dataset.dataset_path=$final_merged_path_depth_seg \
                multi_run.run_dir=$policy_train_save_dir \
                logging.name=${wandb_name} \
                hydra.run.dir=$policy_train_save_dir \
                hydra.sweep.dir=$policy_train_save_dir \
                training.num_epochs=250 \
                policy.obs_encoder.norm_args.agentview_depth.clamp_max=${agentview_depth_clamp_max} \
                policy.obs_encoder.norm_args.agentview_depth.mean=${agentview_depth_mean} \
                policy.obs_encoder.norm_args.agentview_depth.std=${agentview_depth_std}
        fi
    fi
fi

if false; then
    echo "Dummy training configs"
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.:../robosuite HYDRA_FULL_ERROR=1 python train.py --config-name=train_diffusion_unet_ddpm_image_workspace_robomimic \
        task=three_piece_assembly_image_abs task.dataset_type=ph \
        task.dataset_path=/home/thankyou/autom/demo-aug/datasets/generated/three_piece_assembly/2024-11-23-18-55-44/merged_E7.0-100-near-constraint_obs.hdf5 \
        task.env_runner.dataset_path=/home/thankyou/autom/demo-aug/datasets/generated/three_piece_assembly/2024-11-23-18-55-44/merged_E7.0-100-near-constraint_obs.hdf5 \
        task.dataset.dataset_path=/home/thankyou/autom/demo-aug/datasets/generated/three_piece_assembly/2024-11-23-18-55-44/merged_E7.0-100-near-constraint_obs.hdf5
fi
