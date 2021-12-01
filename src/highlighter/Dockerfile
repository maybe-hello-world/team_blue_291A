FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive

RUN apt update && \
    apt-get install -y libopencv-dev python3-opencv ca-certificates python3-dev git wget sudo ninja-build && \
    mkdir /app && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install -r /app/prerequirements.txt

RUN pip install -r /app/requirements.txt