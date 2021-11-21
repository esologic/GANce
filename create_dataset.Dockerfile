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

# To actually execute this, use `sudo docker-compose build create_dataset &&
# sudo docker-compose run create_dataset`.
# The directories `/home/root/stylegan2/`, `/home/root/faces/datasets/`, and
# `/home/root/faces/images/` need to be mapped in with a bind volume mount.
CMD [ \
    "python", \
    "/home/root/stylegan2/dataset_tool.py", \
    "create_from_images", \
    "/home/root/faces/datasets", \
    "/home/root/faces/images"\
]
