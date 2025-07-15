#!/bin/bash

python scripts/segment_images.py \
    --save-as-blender False \
    --segmentation-phrase "the grey robot" \
    --src-json-path data/robomimic/square/newest/85/transforms.json \
    --dest-json-name transforms_seg.json

# python scripts/segment_images.py \
#     --save-as-blender True \
#     --segmentation-phrase "the grey robot" \
#     --src-json-path data/robomimic/lift/newest/timestep_5/transforms_val_src.json \
#     --dest-json-name transforms_val.json

# python scripts/segment_images.py \
#     --save-as-blender True \
#     --segmentation-phrase "the red object" \
#     --src-json-path data/robomimic/lift/newest/timestep_5/transforms_train_src.json \
#     --dest-json-name transforms_train.json

# python scripts/segment_images.py \
#     --save-as-blender True \
#     --segmentation-phrase "the red object" \
#     --src-json-path data/robomimic/lift/newest/timestep_5/transforms_val_src.json \
#     --dest-json-name transforms_val.json
