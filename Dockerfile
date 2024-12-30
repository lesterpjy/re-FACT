FROM continuumio/miniconda3

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV TORCH_HOME=/local/cache
ENV ALLENNLP_CAHCE_ROOT=/local/cache

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    zip \
    gzip \
    make \
    automake \
    gcc \
    build-essential \
    g++ \
    cpp \
    libc6-dev \
    unzip \
    nano \
    rsync

RUN conda update -q conda

RUN mkdir -pv /local/src
RUN mkdir -pv /local/configs
RUN mkdir -pv /local/scripts

VOLUME /local/work
VOLUME /local/cache

WORKDIR /local/
ADD requirements.txt /local/
RUN pip install -r requirements.txt

ADD src /local/src
ADD configs /local/configs
ADD scripts /local/scripts

ENV PYTHONPATH=/local/src
