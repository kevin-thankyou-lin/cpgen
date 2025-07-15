#!/bin/bash


# target: [-0.11118988  0.17443746  0.89      ]
# bbox_min: [-0.15042321  0.13700605  0.88      ]
# bbox_max: [-0.07195655  0.21186886  0.9       ]
# array([0.03923333, 0.03743141, 0.01      ])

PYTHONPATH=. ns-viewer --load-config nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/single-img-pc/nerfacto/2024-07-05_003114/config.yml

# nerf_outputs/nerfacto-data/robomimic/square/newest/85/nerfacto/2023-08-20_181806/config.yml
# PYTHONPATH=. ns-viewer --load-config nerf_outputs/nerfacto-data/robomimic/square/newest0/85/nerfacto/2023-08-20_154114/config.yml


# PYTHONPATH=. ns-viewer --load-config nerf_outputs/nerfacto-data/robomimic/square/newest/50_51/nerfacto/2023-08-20_135113/config.yml
# PYTHONPATH=. ns-viewer --load-config nerf_outputs/nerfacto-data/robomimic/square/newest/50_51/nerfacto/2023-08-20_132607/config.yml
# PYTHONPATH=. ns-viewer --load-config outputs/nerfacto-lift-08-05-16-pitcher-images-ns032/nerfacto/2023-08-05_121449/config.yml
# PYTHONPATH=. ns-viewer --load-config outputs/nerfacto-lift-08-04-ns032/nerfacto/2023-08-04_132249/config.yml
# PYTHONPATH=. ns-viewer --load-config outputs/nerfacto-door-07-28-ns032/nerfacto/2023-07-28_151149/config.yml
# PYTHONPATH=. ns-viewer --load-config outputs/nerfacto-door-07-27/nerfacto/2023-07-27_191549/config.yml
# PYTHONPATH=. ns-viewer --load-config outputs/nerfacto-door-07-26/nerfacto/2023-07-26_233349/config.yml

# need python path if config file is in demo-aug
# PYTHONPATH=. ns-viewer --load-config outputs/nerfacto-lift-06-09/nerfacto/2023-06-09_155956/config.yml
# PYTHONPATH=. ns-viewer --load-config outputs/tensorf-lift-06-22-timestep-5/tensorf/2023-06-22_164049/config.yml
