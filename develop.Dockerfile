# We need to start with this image to be able to get `nvcc`
# You need to be logged into the container as root to be able to use `nvcc` commands.
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install openssh-server sudo -y

# SSH Config
RUN useradd -rm -d /home/gpu -s /bin/bash -g root -G sudo -u 1000 gpu
RUN  echo 'gpu:password' | chpasswd
RUN service ssh start
EXPOSE 22

RUN mkdir -m 777 /home/gpu/yellow_album_cover
COPY ./ /home/gpu/yellow_album_cover
RUN apt-get update && apt-get install sudo -y && \
chmod +x /home/gpu/yellow_album_cover/tools/create_venv_gpu.sh && \
./home/gpu/yellow_album_cover/tools/create_venv_gpu.sh

RUN echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> /home/gpu/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' \
 >> /home/gpu/.bashrc

# Entry Point
# To actually execute this, use `sudo docker-compose up -d --build develop`.
# This will let you SSH into the container and run/change stuff.
# To access the terminal inside the container on the host, use:
# `sudo docker exec -ti develop /bin/bash`
CMD ["/usr/sbin/sshd","-D"]
