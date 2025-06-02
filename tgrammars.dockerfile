# Base image must at least have CUDA installed.

# https://cloud.google.com/deep-learning-containers/docs/choosing-container
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu113.py37

# Install additional programs
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    pkg-config \
    libgoogle-perftools-dev \
    ca-certificates \
    git \
    vim && \
    rm -rf /var/lib/apt/lists

# Update pip
RUN python3 -m pip install --upgrade pip

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install TG dependencies
WORKDIR /
RUN git clone https://github.com/justeuer/transformer_grammars.git
WORKDIR /transformer_grammars
RUN git checkout v1.0.1
RUN mkdir .dependencies 
WORKDIR /transformer_grammars/.dependencies
RUN git clone -b 20220623.1 https://github.com/abseil/abseil-cpp.git
RUN git clone -b 3.4.0 https://gitlab.com/libeigen/eigen.git
RUN git clone -b v2.10.2 https://github.com/pybind/pybind11.git
RUN git clone -b v0.1.97 https://github.com/google/sentencepiece.git
WORKDIR /transformer_grammars/.dependencies/sentencepiece
RUN mkdir build
WORKDIR /transformer_grammars/.dependencies/sentencepiece/build
RUN cmake ..
RUN make -j
RUN make install
RUN ldconfig

# Install TG and python dependencies
WORKDIR /transformer_grammars
RUN pip install --no-cache-dir --upgrade wandb==0.17.4
RUN pip install --no-cache-dir --require-hashes -r requirements.txt
RUN pip install -e . --no-dependencies --no-index

# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME
# this will fix a wandb issue
RUN mkdir /home/$USER_NAME/.local
# Change owner of home dir (Note: this is not the lsv nethome)
RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/
RUN chown -R ${USER_UID}:${USER_GID} /transformer_grammars/
# add user home to python path
ENV PYTHONPATH="/home/$USER_NAME/:$PYTHONPATH"
# change to user
USER $USER_NAME

CMD bash -c "cd /transformer_grammars && git pull && cd - && exec bash"
