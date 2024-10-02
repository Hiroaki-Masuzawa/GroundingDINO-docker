FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
RUN apt update && apt install -y python3 python3-distutils curl git python-is-python3 python3-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl https://bootstrap.pypa.io/get-pip.py | python3
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip cache purge

### speedup download
# https://genzouw.com/entry/2019/09/04/085135/1718/
RUN sed -i 's@archive.ubuntu.com@ftp.jaist.ac.jp/pub/Linux@g' /etc/apt/sources.list
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9" \
    SETUPTOOLS_USE_DISTUTILS=stdlib
    
RUN mkdir /source && cd /source && \
    git clone https://github.com/IDEA-Research/GroundingDINO.git && \
    cd GroundingDINO && pip install -q -e .
RUN pip install -q roboflow
RUN apt update && apt install -y wget && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir /source/weights && cd /source/weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
RUN mkdir /source/data && cd /source/data && \
    wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg && \
    wget -q https://media.roboflow.com/notebooks/examples/dog-2.jpeg && \
    wget -q https://media.roboflow.com/notebooks/examples/dog-3.jpeg && \
    wget -q https://media.roboflow.com/notebooks/examples/dog-4.jpeg
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y  libopencv* && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
    