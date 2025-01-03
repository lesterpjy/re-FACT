FROM lesterpjy10/base-image

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
