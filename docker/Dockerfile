FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 
#ubuntu:16.04

RUN apt update
RUN apt -y upgrade
RUN apt install 
# ========== Mujoco dependencies ========== (https://github.com/openai/mujoco-py/blob/master/Dockerfile)
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ========== Python 3.6 ==========
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3.6 python3-pip
RUN virtualenv --python=python3.6 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

# ========== Mujoco dependency ==========
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

# At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*
ENV LANG C.UTF-8

# ========== Mujoco ========== 
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip

COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

#COPY mujoco-py/vendor/Xdummy /usr/local/bin/Xdummy
#RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY mujoco-py/vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY mujoco-py/requirements.txt /mujoco_py/
COPY mujoco-py/requirements.dev.txt /mujoco_py/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.dev.txt

# Delay moving in the entire code until the very end.
#ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
#CMD ["pytest"]
COPY mujoco-py/. /mujoco_py
RUN python setup.py install

# ========== Anaconda ==========
# https://github.com/ContinuumIO/docker-images/blob/master/anaconda/Dockerfile
#RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
#    libglib2.0-0 libxext6 libsm6 libxrender1 \
#    git mercurial subversion
#RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#    wget  --no-check-certificate --quiet https://repo.continuum.io/archive/Anaconda2-2.5.0-Linux-x86_64.sh && \
#    /bin/bash /Anaconda2-2.5.0-Linux-x86_64.sh -b -p /opt/conda && \
#    rm /Anaconda2-2.5.0-Linux-x86_64.sh

#RUN apt-get install -y curl grep sed dpkg && \
#    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
#    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
#    dpkg -i tini.deb && \
#    rm tini.deb && \
#    apt-get clean

#ENV PATH /opt/conda/bin:$PATH
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
#ENV LANG C.UTF-8
#ENTRYPOINT [ "/usr/bin/tini", "--" ]

# ========== Special Deps ==========
RUN apt-get -y install git make cmake unzip
RUN pip install awscli
# ALE requires zlib
RUN apt-get -y install zlib1g-dev
# Enable installing with build-dep
RUN sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list
RUN apt-get update
# MUJOCO requires graphics stuff (Why?)
RUN apt-get -y build-dep glfw
RUN apt-get -y install libxrandr2 libxinerama-dev libxi6 libxcursor-dev
# copied from requirements.txt
#RUN pip install imageio tabulate nose
RUN apt-get install -y vim ack-grep
RUN pip install --upgrade pip
# usual pip install pygame will fail
RUN apt-get build-dep -y python-pygame
RUN pip install Pillow

# ========== OpenAI Gym ==========
RUN apt-get -y install libgtk2.0-0
RUN pip install gym
#RUN apt-get -y install ffmpeg
RUN apt-get -y install libav-tools
CMD alias ffmpeg="avconv"

# ========== Mujoco for VIME ==========
#RUN pip install 'mujoco-py=1.5'


# ========== Add codebase stub ==========
#CMD mkdir /root/code
#ADD environment.yml /root/code/environment.yml
#RUN conda env create -f /root/code/environment.yml

#ENV PYTHONPATH /root/code/rllab:$PYTHONPATH
#ENV PATH /opt/conda/envs/rllab3/bin:$PATH
#RUN echo "source activate rllab3" >> /root/.bashrc
#ENV BASH_ENV /root/.bashrc
#WORKDIR /root/code

# ========== VIME deps ==========
RUN apt-get -y install swig 
RUN pip install joblib \
    path.py \
    pygame \
    Box2D \
    mako \
    cached_property \
    lasagne \
    pyprind \
    tensorboardX \
    theano \
    jupyter

RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip # (https://github.com/aigamedev/scikit-neuralnetwork/issues/235)

# ========== GUI access ==========
RUN apt-get -y install iputils-ping eog nano  

# ========== Misc ==========
# Import user environment variable
ARG user
ENV USER $user

# gpu theanno
ENV THEANO_FLAGS mode=FAST_RUN,device=gpu,floatX=float32

## ========== Other stuff for Mujoco ==========
# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
#COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

#WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
#COPY ./requirements.txt /mujoco_py/
#COPY ./requirements.dev.txt /mujoco_py/
#RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.dev.txt

# Delay moving in the entire code until the very end.
#ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
#CMD ["pytest"]
#COPY . /mujoco_py
#RUN python setup.py install

#ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
COPY ./docker_entrypoint.sh /
ENTRYPOINT ["sh","/docker_entrypoint.sh"]
CMD ["bash"]
