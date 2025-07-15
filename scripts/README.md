# Scripts

Useful scripts for various functionality (documented below).


# Generating data from real world demo for policy training

1. Generate data: `scripts/real_world/launch_generate_aug_real.sh`
2. Merge data: `scripts/merge_demo_hdf5s.py`
3. Resize/rename keys: `scripts/dataset/process_dataset.sh`

## Visualizing robot end effector position distribution w.r.t target position
Ideally visualize the full pose distribution, but tricky to do so

1. `scripts/data_analysis/plot_eef_pos_relative_to_final.py`


## Evaluating trained policy

1. `scripts/policy/eval_policy.py`
