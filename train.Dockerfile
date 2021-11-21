# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

# There's a warning message that occurrs in 1.15 that states that multi-gpu
# support is best in 1.14
FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN pip install scipy==1.3.3 requests==2.22.0 Pillow==6.2.1

COPY ./yellow_album_cover/stylegan2 /home/root/stylegan2

# To actually execute this, use `sudo docker-compose build train && sudo docker-compose run train`.
# The directories `/home/root/stylegan2/`, `/home/root/faces/results/`, and
# `/home/root/faces/datasets/` need to be mapped in with a bind volume mount. The dataset
# (which is a folder of .tfrecords) `luke` needs to be in the `/home/root/faces/datasets/`
# `directory`.
CMD [ \
    "python", \
    "/home/root/stylegan2/run_training.py", \
    "--num-gpus=1", \
    "--result-dir=/home/root/faces/results/", \
    "--data-dir=/home/root/faces/datasets/", \
    "--dataset=luke", \
    "--total-kimg=1000000", \
    "--config=config-e" \
]
