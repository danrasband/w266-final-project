FROM ubuntu:18.04

# Ignore APT warnings about not having a TTY
ENV DEBIAN_FRONTEND noninteractive

# Ensure UTF-8 locale
RUN apt-get update -qq && \
    apt-get install -y locales && \
    rm -rf /var/lib/apt/lists/*
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && dpkg-reconfigure locales

RUN apt-get update -qq && apt-get upgrade -y  && rm -rf /var/lib/apt/lists/*

# Install dependencies.
RUN apt-get update -qq \
    && apt-get install -y wget git curl build-essential g++ gcc make \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /worker
WORKDIR /worker

ENV ANACONDA_INSTALLER Anaconda3-5.2.0-Linux-x86_64.sh
RUN wget "https://repo.continuum.io/archive/$ANACONDA_INSTALLER"
RUN printf '\nyes\n\nyes\nno\nyes\nno\n' | bash "$ANACONDA_INSTALLER"
RUN /bin/bash -c "source $HOME/.bashrc"

RUN $HOME/anaconda3/bin/pip install --upgrade pip

RUN $HOME/anaconda3/bin/pip install --upgrade \
    boto3 \
    dill \
    networkx \
    numpy \
    pandas

RUN $HOME/anaconda3/bin/pip install spacy==2.0.12
RUN $HOME/anaconda3/bin/pip install \
    https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_lg-3.0.0/en_coref_lg-3.0.0.tar.gz
# RUN $HOME/anaconda3/bin/python -m spacy download en_core_web_lg
RUN $HOME/anaconda3/bin/pip install "msgpack<0.6.0"

RUN mkdir /worker/data
COPY ./data/LP_NET_Graphs.20181205.pkl ./data/documents_dev.pkl /worker/data/
COPY make_predictions.py /worker
