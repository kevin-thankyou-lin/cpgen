POLICY_PATH=/scr/thankyou/autom/consistency-policy/outputs/ddim/wine_glass_grasping/checkpoints/latest.ckpt
POLICY_PATH="/scr/thankyou/autom/consistency-policy/outputs/ddim/wine_glass_grasping/2024-08-02/checkpoints/epoch=0550-train_action_mse_error=0.004.ckpt"
POLICY_PATH="/scr/thankyou/autom/consistency-policy/outputs/ddim/wine_glass_grasping/2024-08-13/checkpoints/epoch=950-train-mse=0.0007.ckpt"
# POLICY_PATH="/scr/thankyou/autom/consistency-policy/outputs/ddim/wine_glass_grasping/2024-08-12/checkpoints/epoch-650-train-mse=0.0011.ckpt"
# POLICY_PATH=/scr/thankyou/autom/consistency-policy/outputs/ddim/wine_glass_grasping/2024-07-28/checkpoints/epoch\=0250-train_action_mse_error\=0.003.ckpt
POLICY_PATH="/scr/thankyou/autom/consistency-policy/outputs/ddim/wine_glass_grasping/2024-08-16/checkpoints/epoch=550-mse-error=0.002.ckpt"
DEMO_PATH=/scr/thankyou/autom/consistency-policy/data/real_world/datasets/augmented/wine_glass_hanging/grasp/narrow/demo_025trials/2024-07-20/nullwine_glass_hanging_grasp_narrow_trials25_se3aug-dx-0.05-0.05-dy-0.05-0.05-dz0.0-0.0-dthetz0.0-0.0-biassampzrot_staug_joint-jointqposnoise0.1_defcurobocfg_CUROBO_1j9so2i4_200demos_resized_renamed.hdf5
GT_SRC_PATH=data/real_world_recordings/inference_data/inference_data_reset_qpos_wine_obj_no_transf.pkl
PYTHONPATH=.:../R2D2/:../consistency-policy/ python scripts/policy/eval_policy.py \
--cfg.policy-path $POLICY_PATH \
--cfg.no-compare-to-ground-truth \
--cfg.demo-aug-cfg.demo-path $DEMO_PATH \
--cfg.ground-truth-src-path $GT_SRC_PATH \
--cfg.ood_variant object_color \
--cfg.obj-transf-variant grid_sweep \
--cfg.bg-shift-type fixed \
--cfg.target-object wine_glass \
--cfg.demo-aug-cfg.task-name wine_glass_hanging \
--cfg.demo-aug-cfg.subtask-name grasp \
--cfg.demo-aug-cfg.aug-cfg.se3-aug-cfg.dx-range -0.05 0.05 \
--cfg.demo-aug-cfg.aug-cfg.se3-aug-cfg.dy-range -0.05 0.05 \
--cfg.demo-aug-cfg.aug-cfg.se3-aug-cfg.dz-range 0 0 \
--cfg.demo-aug-cfg.aug-cfg.se3-aug-cfg.dthetaz-range 0 0 \
--cfg.save-dir policy_eval_outputs_jnt_and_near_goal_start_policy_no_ik_qpos_n_act_8_grid_sweep \
fr3-real-env \
--motion-planner-cfg.motion-planner-type CUROBO


# TODO(klin): actually eval both policies on randomly generated obj trans params
# --cfg.save-dir policy_eval_outputs_jnt_cart_merge_policy_no_ik_qpos_n_act_8 \
