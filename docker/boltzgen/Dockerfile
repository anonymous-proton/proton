# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 \
    HF_HOME=/cache
    
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    libgl1 \
    libhdf5-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3.10 -m pip install --upgrade pip setuptools setuptools_scm wheel

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -e /app

ARG DOWNLOAD_WEIGHTS=false
RUN echo "DOWNLOAD_WEIGHTS=${DOWNLOAD_WEIGHTS}" && \
    if [ "${DOWNLOAD_WEIGHTS}" = "true" ]; then \
        echo "Running boltzgen download..."; \
        boltzgen download all --cache "${HF_HOME}" --force_download; \
    else \
        echo "Skipping download"; \
    fi




WORKDIR /workspace

ENTRYPOINT []
CMD ["/bin/bash"]