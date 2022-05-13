# We need to start with this image to be able to get `nvcc`
# You need to be logged into the container as root to be able to use `nvcc` commands.
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# TODO - we should be able to drop this:
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Requirements to get the container set up, requirements for running the application are installed
# with `create_venv.sh`.
RUN apt-get update && apt-get install openssh-server sudo -y

# User Config
RUN useradd -rm -d /home/gpu -s /bin/bash -g root -G sudo -u 1000 gpu
RUN  echo 'gpu:password' | chpasswd

# SSH Config
RUN service ssh start
EXPOSE 22

RUN mkdir -m 777 /home/gpu/gance
COPY ./ /home/gpu/gance
RUN apt-get update && apt-get install sudo -y && \
chmod +x /home/gpu/gance/tools/create_venv.sh && \
./home/gpu/gance/tools/create_venv.sh

RUN echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> /home/gpu/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' \
 >> /home/gpu/.bashrc

# Entry Point
# To actually execute this, use `sudo docker-compose up -d --build gance-develop`.
# This will let you SSH into the container and run/change stuff.
# To access the terminal inside the container on the host, use:
# `sudo docker exec -ti develop /bin/bash`
CMD ["/usr/sbin/sshd","-D"]
