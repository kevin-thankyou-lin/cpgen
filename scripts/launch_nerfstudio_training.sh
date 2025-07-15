#!/bin/bash

ns-train splatfacto --pipeline.model.sh-degree 1 nerfstudio-data  --data /scr/thankyou/autom/demo-aug/data/real_world_recordings/agent_w_hand_views_2024-07-06-23:44:47/ --auto_scale_poses False --orientation-method none --center-method none
exit


ns-train splatfacto --pipeline.model.sh-degree 0 nerfstudio-data  --data /scr/thankyou/autom/demo-aug/data/real_world_recordings/background_scan_2024-07-05-15:42:32/ --auto_scale_poses False --orientation-method none --center-method none
exit
ns-train nerfacto \
nerfstudio-data  --data /scr/thankyou/autom/demo-aug/data/real_world_recordings/single-img-pc/ --auto_scale_poses False --orientation-method none --center-method none



# random scale needs to be large enough to cover the cameras and a bit more, I think
ns-train splatfacto \
--pipeline.model.random-scale 3.0  \
nerfstudio-data  --data /scr/thankyou/autom/demo-aug/data/real_world_recordings/single-img-pc/ --auto_scale_poses False --orientation-method none --center-method none

# nerfstudio-data  --data data/robomimic/square/newest/31/ --auto_scale_poses False --orientation-method none --center-method none

ns-train nerfacto \
--pipeline.model.proposal-initial-sampler uniform \
--pipeline.model.camera-optimizer.mode off \
--pipeline.model.near-plane 0.01 \
--pipeline.model.far-plane 3. \
--pipeline.model.use-average-appearance-embedding False \
--pipeline.model.distortion-loss-mult 0 \
--pipeline.model.disable-scene-contraction False \
--pipeline.model.predict-normals True \
--data=data/robomimic/square/newest/39 \
--experiment-name=nerfacto-v1.1-square-39-25-03-no-appear-embed--no-scale-or-orient-or-center \
--relative-model-dir=nerfstudio_models/ \
--steps-per-save=1000 \
--max-num-iterations=2500 \
nerfstudio-data --auto_scale_poses False --orientation-method none --center-method none \
# --vis wandb nerfstudio-data --auto_scale_poses False --orientation-method none --center-method none \
# --pipeline.model.background-color white \
# --pipeline.datamanager.camera-optimizer.mode off \
# --experiment-name=nerfacto-v034-square-35-25-03-no-appear-embed--no-scale-or-orient-or-center \

# depth-nerfacto works
# ns-train depth-nerfacto \
# --pipeline.model.background-color white \
# --pipeline.model.proposal-initial-sampler uniform \
# --pipeline.datamanager.camera-optimizer.mode off \
# --pipeline.model.near-plane 0.02 \
# --pipeline.model.far-plane 2. \
# --pipeline.model.use-average-appearance-embedding False \
# --pipeline.model.distortion-loss-mult 0 \
# --pipeline.model.disable-scene-contraction False \
# --pipeline.model.predict-normals True \
# --pipeline.model.depth-loss-mult 0.01 \
# --pipeline.model.depth-loss-type SPARSENERF_RANKING \
# --pipeline.datamanager.pixel-sampler.radius 30 \
# --data=data/robomimic/square/newest/35 \
# --experiment-name=nerfacto-v034-square-35-24-03-depthloss1e-2-no-appear-embed-pair-pixel-rad30-no-scale-or-orient-or-center \
# --relative-model-dir=nerfstudio_models/ \
# --steps-per-save=1000 \
# --max-num-iterations=2500 \
# --vis wandb nerfstudio-data --auto_scale_poses False --orientation-method none --center-method none \

# ns-train nerfacto \
# --pipeline.model.background-color white \
# --pipeline.model.proposal-initial-sampler uniform \
# --pipeline.datamanager.camera-optimizer.mode off \
# --pipeline.model.near-plane 0.02 \
# --pipeline.model.far-plane 4. \
# --pipeline.model.use-average-appearance-embedding False \
# --pipeline.model.distortion-loss-mult 0 \
# --pipeline.model.disable-scene-contraction False \
# --pipeline.model.predict-normals True \
# --data=data/robomimic/lift/newest/timestep_8 \
# --experiment-name=nerfacto-lift-08-05-16-pitcher-images-ns032 \
# --relative-model-dir=nerfstudio_models/ \
# --steps-per-save=1000 \
# --max-num-iterations=16500 \
# --vis wandb \
# --timestamp 2023-08-05_121449 blender-data

# ns-train nerfacto \
# --pipeline.model.background-color white \
# --pipeline.model.proposal-initial-sampler uniform \
# --pipeline.datamanager.camera-optimizer.mode off \
# --pipeline.model.near-plane 0.02 \
# --pipeline.model.far-plane 3. \
# --pipeline.model.use-average-appearance-embedding False \
# --pipeline.model.distortion-loss-mult 0 \
# --pipeline.model.disable-scene-contraction False \
# --pipeline.model.predict-normals True \
# --data=data/robomimic/lift/newest/timestep_8 \
# --experiment-name=nerfacto-lift-08-04-12-images-ns032 \
# --relative-model-dir=nerfstudio_models/ \
# --steps-per-save=1000 \
# --max-num-iterations=16500 \
# --vis wandb \
# --timestamp 2023-08-04_134649 blender-data


# ns-train tensorf \
# --data=data/robomimic/lift/newest/timestep_5 \
# --experiment-name=tensorf-lift-06-22-timestep-5 \
# --relative-model-dir=nerfstudio_models/ \
# --steps-per-save=1000 \
# --max-num-iterations=16500 \
# --vis wandb \
# --timestamp 2023-06-22_164049 blender-data

# ns-train nerfacto \
# --pipeline.model.background-color white \
# --pipeline.model.proposal-initial-sampler uniform \
# --pipeline.datamanager.camera-optimizer.mode off \
# --pipeline.model.near-plane 0.02 \
# --pipeline.model.far-plane 3. \
# --pipeline.model.use-average-appearance-embedding False \
# --pipeline.model.distortion-loss-mult 0 \
# --pipeline.model.disable-scene-contraction False \
# --pipeline.model.predict-normals True \
# --data=data/robomimic/door/newest-imagehdf5-correct-reset/timestep_99 \
# --experiment-name=nerfacto-door-07-28-ns032 \
# --relative-model-dir=nerfstudio_models/ \
# --steps-per-save=1000 \
# --max-num-iterations=16500 \
# --vis wandb \
# --timestamp 2023-07-28_151149 blender-data

# --timestamp 2023-07-26_233349 blender-data

# ns-train tensorf \
# --data=data/robomimic/lift/newest/timestep_51 \
# --experiment-name=lift_2023-06-09 \
# --relative-model-dir=nerfstudio_models/ \
# --steps-per-save=1000 \
# --max-num-iterations=16500 \
# --vis wandb \
# --timestamp 2023-06-09_154601 blender-data

# ns-train nerfacto \
# --pipeline.model.background-color white \
# --pipeline.model.proposal-initial-sampler uniform \
# --pipeline.model.use-average-appearance-embedding False \
# --pipeline.model.distortion-loss-mult 0 \
# --data=data/robomimic/lift/new/timestep_51 \
# --experiment-name=lift_2023-06-09 \
# --relative-model-dir=nerfstudio_models/ \
# --steps-per-save=1000 \
# --max-num-iterations=16500 \
# --vis wandb \
# --timestamp 2023-06_09_152854 blender-data

# ns-train nerfacto \
# --pipeline.model.background-color white \
# --pipeline.model.proposal-initial-sampler uniform \
# --pipeline.datamanager.camera-optimizer.mode off \
# --pipeline.model.near-plane 0.02 \
# --pipeline.model.far-plane 6. \
# --pipeline.model.use-average-appearance-embedding False \
# --pipeline.model.distortion-loss-mult 0 \
# --pipeline.model.disable-scene-contraction False \
# --data=data/robomimic/lift/new/timestep_51 \
# --experiment-name=nerfacto-lift-06-09 \
# --relative-model-dir=nerfstudio_models/ \
# --steps-per-save=1000 \
# --max-num-iterations=16500 \
# --vis wandb \
# --timestamp 2023-06-09_155956 blender-data
