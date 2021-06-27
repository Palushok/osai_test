FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04-rc
WORKDIR /app
COPY requirements.txt /app/.

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3.8 python3-pip python3.8-dev -y && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \ 
    python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install --upgrade setuptools && \
    python3.8 -m pip install --no-cache-dir -I -r requirements.txt && \
    apt-get install ffmpeg libsm6 libxext6  -y

COPY . /app/.