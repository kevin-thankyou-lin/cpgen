#!/bin/bash

# Usage: ./launch_generate_aug.sh <seed_start> <seed_end> <task_name> <subtask_name> <task_distrbution> <trials>
# where trials = expected number of aug'ed demos per constraint, for a given seed

# randomize task-relevant object SE3: wp2goal
do_start_at_wp=1
use_mjc_rend=1
do_wp2goal=0
do_fixedwp2fixednewgoal=0
do_fixedwp2fixednewgoal_usegtobjrend=0
task="$3"
subtask="$4"
task_distribution="$5"  # narrow or wide or narrow-debug
trials="$6"

method="None"  # or null

start_distribution="near_retract" # near_goal or near_retract
MOTION_PLANNER="CUROBO"
# start_distribution="near_goal" # near_goal or near_retract
# MOTION_PLANNER="PRM"

# assert task is one of door, lift, can, square
if [ "$task" != "door" ] && [ "$task" != "lift" ] && [ "$task" != "can" ] && [ "$task" != "square" ]; then
    echo "Invalid task: $task. Must be door, lift, can, or square."
    echo "Correct format is ./launch_generate_aug.sh <seed_start> <seed_end> <task_name> <subtask_name> <task_distrbution> <trials>"
    echo "Got: $@"
    exit 1
fi

case "$task_distribution" in
    narrow|wide|narrow-debug|narrow-debug-w-scale|narrow-debug-w-scale-opp-rot|wide-figures|single)
        ;;
    *)
        echo "Invalid task distribution: $task_distribution. Must be narrow or wide or narrow-debug."
        echo "Correct format is ./launch_generate_aug.sh <seed_start> <seed_end> <task_name> <subtask_name> <task_distrbution> <trials>"
        echo "Got: $@"
        exit 1
        ;;
esac

collision_free_wp_threshold=0.07
save_base_dir="/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/${task}/${subtask}/${task_distribution}/"

use_wandb=1
view_meshcat=0


set_task_config() {
    case $task in
        door)
            demo_path="../diffusion_policy/data/robomimic/datasets/door/subsampled_image.hdf5"
            eef_rot_angle_bound=0.85
            ;;
        lift)
            demo_path="../diffusion_policy/data/robomimic/datasets/lift/ph/1_demo.hdf5"
            eef_rot_angle_bound=0.85
            ;;
        can)
            # demo_path="../diffusion_policy/data/robomimic/datasets/can/ph/1_demo.hdf5"
            # demo_path="../diffusion_policy/data/robomimic/datasets/can/ph/1_demo_from_robomimic.hdf5"
            demo_path="../diffusion_policy/data/robomimic/datasets/can/ph/1_demo_from_robomimic_sep5.hdf5"
            eef_rot_angle_bound=0.85
            if [ "$subtask" != "grasp" ] && [ "$subtask" != "drop" ]; then
                echo "Invalid subtask: $subtask. Must be grasp or drop for can task."
                exit 1
            fi
            ;;
        square)
            demo_path="../diffusion_policy/data/robomimic/datasets/square/ph/1_demo.hdf5"
            eef_rot_angle_bound=1.6
            if [ "$subtask" != "grasp" ] && [ "$subtask" != "insert" ]; then
                echo "Invalid subtask: $subtask. Must be grasp or insert for square task."
                exit 1
            fi
            ;;
        *)
            echo "Invalid task: $task"
            exit 1
            ;;
    esac
}

build_command() {
    common_params="--demo-aug-cfg.trials-per-constraint $trials \
                   --demo-aug-cfg.demo-path \"$demo_path\" \
                   --demo-aug-cfg.save-base-dir $save_base_dir \
                   --demo-aug-cfg.task-name $task \
                   --demo-aug-cfg.subtask-name $subtask \
                   --demo-aug-cfg.task_distribution $task_distribution \
                   --demo-aug-cfg.seed $seed"

    if [ "$method" != "None" ]; then
        common_params="$common_params --demo-aug-cfg.method $method"
    fi

    get_lift_params() {
        subtask_distribution_common_params="--demo-aug-cfg.aug-cfg.init-gripper-type non-fixed \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.use-abs-transform \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.use-biased-sampling-z-rot \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dthetaz-range 2.3 3.9 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dz-range 0.8211 0.8211 \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.space joint \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.joint-space-aug-configs.joint-qpos-noise-magnitude 0.06 \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.end-with-tracking-start-gripper-qpos"
        if [ "$task_distribution" = "narrow" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-scale-aug \
                --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range -0.03 0.03 \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range -0.03 0.03"
        elif [ "$task_distribution" = "wide" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.apply-scale-aug \
                --demo-aug-cfg.aug-cfg.scale-aug-cfg.apply-non-uniform-scaling \
                --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.7 1.2 \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range -0.1 0.1 \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range -0.1 0.1"
        elif [ "$task_distribution" = "wide-figures" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.no-apply-non-uniform-scaling \
                  --demo-aug-cfg.aug-cfg.no-apply-ee-noise-aug \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 1.05 1.05"
        else
            echo "Invalid task distribution: $task_distribution. Must be narrow or wide."
            exit 1
        fi

        if [ "$method" = "spartan" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-scale-aug \
                --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.no-use-abs-transform \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range 0.0 0.0 \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range 0.0 0.0 \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dz-range 0.0 0.0 \
                --demo-aug-cfg.aug-cfg.start-aug-cfg.space joint \
                --demo-aug-cfg.aug-cfg.start-aug-cfg.joint-space-aug-configs.joint-qpos-noise-magnitude 0.06     \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dthetaz-range 0.0 0.0"
        fi

        echo "$subtask_distribution_common_params $subtask_distribution_specific_params"
    }

    get_square_grasp_params() {
        subtask_distribution_common_params="--demo-aug-cfg.aug-cfg.init-gripper-type non-fixed \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.use-abs-transform \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range -0.2 -0.07 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range 0.05 0.28 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dz-range 0.8299 0.8299 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dthetaz-range -2.32 3.92 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.use-robosuite-placement-initializer \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.space joint \
            --demo-aug-cfg.aug-cfg.reflect-eef-and-select-min-cost \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.joint-space-aug-configs.joint-qpos-noise-magnitude 0.06 \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.end-with-tracking-start-gripper-qpos"
        if [ "$task_distribution" = "narrow" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-scale-aug \
                  --demo-aug-cfg.aug-cfg.no-apply-shear-aug"
        elif [ "$task_distribution" = "wide" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.apply-scale-aug \
                  --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.8 1.1"
        elif [ "$task_distribution" = "narrow-debug" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.se3-aug-cfg.no-use-robosuite-placement-initializer \
                  --demo-aug-cfg.aug-cfg.no-apply-scale-aug \
                  --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range -0.116 -0.109 \
                  --demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range 0.109 0.23 \
                  --demo-aug-cfg.aug-cfg.se3-aug-cfg.dthetaz-range 1.57 4.712 \
                  --demo-aug-cfg.aug-cfg.no-reflect-eef-and-select-min-cost"
        elif [ "$task_distribution" = "narrow-debug-w-scale" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.se3-aug-cfg.no-use-robosuite-placement-initializer \
                  --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.apply-scale-aug \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.8 1.1 \
                  --demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range -0.116 -0.109 \
                  --demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range 0.109 0.23 \
                  --demo-aug-cfg.aug-cfg.se3-aug-cfg.dthetaz-range 1.57 4.712 \
                  --demo-aug-cfg.aug-cfg.no-reflect-eef-and-select-min-cost"
        elif [ "$task_distribution" = "narrow-debug-w-scale-opp-rot" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.se3-aug-cfg.no-use-robosuite-placement-initializer \
                  --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.apply-scale-aug \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.8 1.1 \
                  --demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range -0.116 -0.109 \
                  --demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range 0.109 0.23 \
                  --demo-aug-cfg.aug-cfg.se3-aug-cfg.dthetaz-range -1.572 1.57 \
                  --demo-aug-cfg.aug-cfg.always-reflect-eef \
                  --demo-aug-cfg.aug-cfg.no-reflect-eef-and-select-min-cost"
        elif [ "$task_distribution" = "single" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.init-gripper-type non-fixed \
            --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
            --demo-aug-cfg.aug-cfg.no-apply-scale-aug \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.use-abs-transform \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range -0.14 -0.14 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range 0.13 0.13 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dz-range 0.8299 0.8299 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dthetaz-range 0 0 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.no-use-robosuite-placement-initializer \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.space joint \
            --demo-aug-cfg.aug-cfg.reflect-eef-and-select-min-cost \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.joint-space-aug-configs.joint-qpos-noise-magnitude 0.06 \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.end-with-tracking-start-gripper-qpos"
        else
            echo "Invalid task distribution: $task_distribution. Must be narrow or wide or narrow-debug."
            exit 1
        fi

        # have a start distribution if statement
        if [ "$start_distribution" = "near_goal" ]; then
            subtask_distribution_specific_params="$subtask_distribution_specific_params \
                --demo-aug-cfg.aug-cfg.start-aug-cfg.space cartesian \
                --demo-aug-cfg.aug-cfg.no-reflect-eef-and-select-min-cost \
                --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-z-bound -30 30 \
                --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-y-bound -12 12 \
                --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-x-bound -12 12 \
                --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-pos-bound 0.05"
        fi

        echo "$subtask_distribution_common_params $subtask_distribution_specific_params"
    }

    get_square_insert_params() {
        subtask_distribution_common_params="--demo-aug-cfg.aug-cfg.se3-aug-cfg.no-use-abs-transform \
                  --demo-aug-cfg.aug-cfg.no-apply-se3-aug \
                  --demo-aug-cfg.aug-cfg.apply-ee-noise-aug \
                  --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-x-min -0.01 \
                  --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-x-max 0.025 \
                  --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-y-min -0.01 \
                  --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-y-max 0.01 \
                  --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-z-min -0.02 \
                  --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-z-max 0.02 \
                  --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.rot-bound 0 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.space cartesian \
                  --demo-aug-cfg.aug-cfg.reflect-eef-and-select-min-cost \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-pos-bound 0.13 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-z-bound -110 110 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-y-bound -22.5 22.5 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-x-bound -22.5 22.5 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.ee-pos -0 0.16 0.82 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.end-with-tracking-start-gripper-qpos"
        if [ "$task_distribution" = "narrow" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-scale-aug \
                  --demo-aug-cfg.aug-cfg.no-apply-ee-noise-aug \
                  --demo-aug-cfg.aug-cfg.no-apply-shear-aug"
        elif [ "$task_distribution" = "wide" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.8 1.1"
        elif [ "$task_distribution" = "wide-figures" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-z-bound -40 40 \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.apply-non-uniform-scaling \
                  --demo-aug-cfg.aug-cfg.no-apply-ee-noise-aug \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.9 1.6"
        elif [ "$task_distribution" = "narrow-debug" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-scale-aug \
                  --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.no-reflect-eef-and-select-min-cost \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-pos-bound 0.16 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.ee-pos -0.05 0.16 0.82 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-z-bound -110 110 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-y-bound -22.5 22.5 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-x-bound -22.5 22.5"
        elif [ "$task_distribution" = "narrow-debug-w-scale" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.apply-scale-aug \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.8 1.1 \
                  --demo-aug-cfg.aug-cfg.no-reflect-eef-and-select-min-cost \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-pos-bound 0.16 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.ee-pos -0.05 0.16 0.82 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-z-bound -110 110 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-y-bound -22.5 22.5 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-x-bound -22.5 22.5"
        elif [ "$task_distribution" = "narrow-debug-w-scale-opp-rot" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.apply-scale-aug \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.8 1.1 \
                  --demo-aug-cfg.aug-cfg.no-reflect-eef-and-select-min-cost \
                  --demo-aug-cfg.aug-cfg.always-reflect-eef \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-pos-bound 0.16 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.ee-pos -0.05 0.16 0.88 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-z-bound -110 110 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-y-bound -22.5 22.5 \
                  --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-x-bound -22.5 22.5"
        else
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.apply-scale-aug \
                  --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                  --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.8 1.1"
        fi
        echo "$subtask_distribution_common_params $subtask_distribution_specific_params"
    }

    # check these dz ranges
    get_can_grasp_params() {
        subtask_distribution_common_params="--demo-aug-cfg.aug-cfg.init-gripper-type non-fixed \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.use-abs-transform \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range -0.05 0.25\
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range -0.45 -0.05 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dz-range 0.86 0.86 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.dthetaz-range -0.7 -0.9 \
            --demo-aug-cfg.aug-cfg.se3-aug-cfg.no-use-robosuite-placement-initializer \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.space joint \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.joint-space-aug-configs.joint-qpos-noise-magnitude 0.06 \
            --demo-aug-cfg.aug-cfg.start-aug-cfg.end-with-tracking-start-gripper-qpos"
        if [ "$task_distribution" = "narrow" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-scale-aug \
                --demo-aug-cfg.aug-cfg.no-apply-shear-aug"
        elif [ "$task_distribution" = "wide" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.apply-scale-aug \
                --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.6 1.2"
        elif [ "$task_distribution" = "single" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-scale-aug \
                --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.no-use-abs-transform \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range 0.0 0.0 \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range 0.0 0.0 \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dz-range 0.0 0.0 \
                --demo-aug-cfg.aug-cfg.start-aug-cfg.space joint \
                --demo-aug-cfg.aug-cfg.start-aug-cfg.joint-space-aug-configs.joint-qpos-noise-magnitude 0.06 \
                --demo-aug-cfg.aug-cfg.se3-aug-cfg.dthetaz-range 0.0 0.0"
        else
            echo "Invalid task distribution: $task_distribution. Must be narrow or wide."
            exit 1
        fi

        echo "$subtask_distribution_common_params $subtask_distribution_specific_params"
    }

    get_can_drop_params() {
        subtask_distribution_common_params="--demo-aug-cfg.aug-cfg.init-gripper-type non-fixed \
        --demo-aug-cfg.aug-cfg.no-apply-shear-aug \
        --demo-aug-cfg.aug-cfg.no-apply-se3-aug \
        --demo-aug-cfg.aug-cfg.apply-ee-noise-aug \
        --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-x-min -0.02 \
        --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-x-max 0.02 \
        --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-y-min -0.02 \
        --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-y-max 0.02 \
        --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-z-min -0.01 \
        --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.pos-z-max 0.01 \
        --demo-aug-cfg.aug-cfg.ee-noise-aug-cfg.rot-bound 0.1 \
        --demo-aug-cfg.aug-cfg.start-aug-cfg.space cartesian \
        --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-pos-bound 0.25 \
        --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-z-bound -22 22 \
        --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-y-bound -22 22 \
        --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-x-bound -22 22 \
        --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.ee-pos 0.1 -0.25 1 \
        --demo-aug-cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.ee-quat-xyzw 0.99971 0.00578 0.02347 0.0014"
        if [ "$task_distribution" = "narrow" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.no-apply-scale-aug"
        elif [ "$task_distribution" = "wide" ]; then
            subtask_distribution_specific_params="--demo-aug-cfg.aug-cfg.apply-scale-aug \
                --demo-aug-cfg.aug-cfg.scale-aug-cfg.scale-factor-range 0.5 0.5"
        else
            echo "Invalid task distribution: $task_distribution. Must be narrow or wide."
            exit 1
        fi

        echo "$subtask_distribution_common_params $subtask_distribution_specific_params"
    }

    case $task in
        lift)
            task_specific_params=$(get_lift_params)
            ;;
        square)
            if [ "$subtask" = "grasp" ]; then
                task_specific_params=$(get_square_grasp_params)
            elif [ "$subtask" = "insert" ]; then
                task_specific_params=$(get_square_insert_params)
            fi
            ;;
        can)
            if [ "$subtask" = "grasp" ]; then
                task_specific_params=$(get_can_grasp_params)
            elif [ "$subtask" = "drop" ]; then
                task_specific_params=$(get_can_drop_params)
            fi
            ;;
        *)
            task_specific_params=""
            ;;
    esac

    [ "$use_wandb" = "1" ] && task_specific_params="$task_specific_params --demo-aug-cfg.use-wandb"
    [ "$view_meshcat" = "1" ] && task_specific_params="$task_specific_params --demo-aug-cfg.aug-cfg.view-meshcat"
    [ "$do_start_at_wp" = "1" ] && task_specific_params="$task_specific_params --demo-aug-cfg.aug-cfg.start-aug-cfg.start-at-collision-free-waypoint"

    env_params=""
    [ "$use_mjc_rend" = "1" ] && env_params="--obs-cfg.use-mujoco-renderer-only"
    [ "$method" = "spartan" ] && env_params="$env_params --motion-planner-cfg.truncate-last-n-steps 6"


    echo "PYTHONPATH=. python scripts/generate_aug.py $common_params $task_specific_params panda-sim-env --motion-planner-cfg.motion-planner-type $MOTION_PLANNER $env_params"
}


set_task_config

for seed in $(seq $1 $2); do
    command=$(build_command)
    eval $command
done
