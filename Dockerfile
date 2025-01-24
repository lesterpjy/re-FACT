FROM lesterpjy10/base-image

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    graphviz-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -pv /local/src /local/configs /local/scripts /local/work /local/cache

COPY requirements.txt /local/
RUN pip install --no-cache-dir -r /local/requirements.txt

COPY src /local/src
COPY configs /local/configs
COPY scripts /local/scripts

ENV PYTHONPATH=/local/src
WORKDIR /local/

VOLUME /local/work
VOLUME /local/cache
