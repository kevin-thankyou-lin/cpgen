#!/bin/bash

ns-export poisson --load-config nerf_outputs/nerfacto-data/robomimic/square/newest/50_51/nerfacto/2023-08-20_135113/config.yml --output-dir exports/mesh/ --target-num-faces 5000 --num-pixels-per-side 1024 \
--normal-method model_output --normal-output-name pred_normals --num-points 10000 --remove-outliers True --use-bounding-box True --bounding-box-min 0.2125 0.0825 0.745 --bounding-box-max 0.2475 0.1175 0.955
# new AABB values from new viewer (?) or me doing stuff w/ new viewer
# although `which ns-export`` doesn't point to the correct pip installed ns-export, we end up using the correct pip installed version
# ns-export poisson --load-config outputs/nerfacto-door-07-28-ns032/nerfacto/2023-07-28_151149/config.yml --output-dir exports/mesh/ --target-num-faces 50000 --num-pixels-per-side 2048 --normal-method model_output --normal-output-name pred_normals --num-points 1000000 --remove-outliers True --use-bounding-box True --bounding-box-min -0.6 -0.6 0.6 --bounding-box-max 0.2 -0.2 1.4
